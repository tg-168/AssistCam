import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

VIDEO_NAME = "00d8944b-e157478b"
GT_CSV_PATH = "mot_labels.csv"

PREDS_BASELINE_PATH = "eval_BASELINE_FULL.jsonl"
PREDS_SKIP_MOTION_PATH = "eval_SKIP_MOTION.jsonl"
PREDS_SKIP_MOTION_TRACK_PATH = "eval_SKIP_MOTION_TRACK.jsonl"

VALID_CLASSES = {"car", "pedestrian", "truck", "bus"}

IOU_THR = 0.2


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def load_predictions(path: str, valid_classes) -> Tuple[
    Dict[int, List[Tuple[float, float, float, float, str, float]]], int
]:
    """
    Read eval_*.jsonl and return:
      - preds_by_frame: frame_idx -> list of (x1, y1, x2, y2, label, score)
      - max_frame_idx
    Ignores any extra keys (inference_ran, is_keyframe, motion_score, etc.).
    """
    label_map = {
        "person": "pedestrian",
        "pedestrian": "pedestrian",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
    }

    preds_by_frame: Dict[int, List[Tuple[float, float, float, float, str, float]]] = defaultdict(list)
    max_frame_idx = -1

    frame_key = None
    dets_key = None
    coord_mode = None
    label_key = None
    score_key = None

    def infer_top_level_keys(rec):
        nonlocal frame_key, dets_key
        if frame_key is None:
            for k, v in rec.items():
                if isinstance(v, int) and any(s in k.lower() for s in ("frame", "idx", "index")):
                    frame_key = k
                    break
        if dets_key is None:
            for k, v in rec.items():
                if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                    dets_key = k
                    break

    def infer_detection_keys(det):
        nonlocal coord_mode, label_key, score_key
        dkeys = set(det.keys())
        if coord_mode is None:
            if "bbox" in dkeys:
                coord_mode = "bbox"
            elif {"x1", "y1", "x2", "y2"}.issubset(dkeys):
                coord_mode = "xyxy"
            elif {"xmin", "ymin", "xmax", "ymax"}.issubset(dkeys):
                coord_mode = "xyminmax"
        if label_key is None:
            for cand in ["category", "label", "class", "name"]:
                if cand in dkeys:
                    label_key = cand
                    break
        if score_key is None:
            for cand in ["score", "confidence", "conf"]:
                if cand in dkeys:
                    score_key = cand
                    break

    total_dets = 0

    if not os.path.exists(path):
        print(f"[WARN] Predictions file not found: {path}")
        return {}, -1

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            infer_top_level_keys(rec)
            if frame_key is None or dets_key is None:
                continue

            try:
                fi = int(rec[frame_key])
            except Exception:
                continue

            max_frame_idx = max(max_frame_idx, fi)

            dets = rec.get(dets_key, [])
            if not isinstance(dets, list):
                continue

            for det in dets:
                if not isinstance(det, dict):
                    continue

                if coord_mode is None or label_key is None:
                    infer_detection_keys(det)

                if coord_mode is None or label_key is None:
                    continue

                try:
                    if coord_mode == "bbox":
                        x1, y1, x2, y2 = det["bbox"]
                    elif coord_mode == "xyxy":
                        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    elif coord_mode == "xyminmax":
                        x1, y1, x2, y2 = det["xmin"], det["ymin"], det["xmax"], det["ymax"]
                    else:
                        continue
                except Exception:
                    continue

                raw_label = det.get(label_key)
                if raw_label is None:
                    continue
                raw_label = str(raw_label)

                mapped_label = label_map.get(raw_label)
                if mapped_label is None:
                    continue

                if mapped_label not in valid_classes:
                    continue

                if score_key is not None and score_key in det:
                    score = float(det[score_key])
                else:
                    score = 1.0

                preds_by_frame[fi].append(
                    (float(x1), float(y1), float(x2), float(y2), mapped_label, score)
                )
                total_dets += 1

    print(
        f"[DEBUG] Parsed predictions from {path}: "
        f"frame_key={frame_key}, dets_key={dets_key}, coord_mode={coord_mode}, "
        f"label_key={label_key}, score_key={score_key}"
    )
    print(f"[DEBUG] Loaded {total_dets} detections in {len(preds_by_frame)} frames from {path}")

    return preds_by_frame, max_frame_idx


def load_ground_truth(csv_path: str, video_name: str, n_frames: int, valid_classes):
    df = pd.read_csv(csv_path, low_memory=False)

    if "videoName" not in df.columns:
        raise KeyError("Expected a 'videoName' column in CSV")

    df = df[df["videoName"] == video_name].copy()
    if df.empty:
        raise ValueError(f"No rows found for videoName={video_name}")

    if "category" in df.columns:
        cat_col = "category"
    elif "label" in df.columns:
        cat_col = "label"
    else:
        raise KeyError("Expected a 'category' or 'label' column in CSV")

    if {"box2d.x1", "box2d.y1", "box2d.x2", "box2d.y2"}.issubset(df.columns):
        x1_col, y1_col, x2_col, y2_col = "box2d.x1", "box2d.y1", "box2d.x2", "box2d.y2"
    elif {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        x1_col, y1_col, x2_col, y2_col = "x1", "y1", "x2", "y2"
    else:
        raise KeyError("Could not find box coordinate columns (x1,y1,x2,y2) in CSV")

    if "frameIndex" not in df.columns:
        raise KeyError("Expected a 'frameIndex' column in CSV")

    df = df[df[cat_col].isin(valid_classes)].copy()
    if df.empty:
        raise ValueError(f"No GT boxes for valid classes {valid_classes} in video {video_name}")

    max_label_idx = int(df["frameIndex"].max())
    scale = n_frames / float(max_label_idx + 1) if max_label_idx > 0 else 1.0
    print(f"[DEBUG] GT: max frameIndex={max_label_idx}, n_frames={n_frames}, scale={scale:.4f}")

    gt_by_frame = defaultdict(list)
    for _, row in df.iterrows():
        raw_idx = int(row["frameIndex"])
        mapped = int(round(raw_idx * scale))
        if mapped < 0 or mapped >= n_frames:
            continue

        x1 = float(row[x1_col]); y1 = float(row[y1_col])
        x2 = float(row[x2_col]); y2 = float(row[y2_col])
        label = str(row[cat_col])
        if label not in valid_classes:
            continue

        gt_by_frame[mapped].append((x1, y1, x2, y2, label))

    total_gt = sum(len(v) for v in gt_by_frame.values())
    print(f"[DEBUG] Loaded {total_gt} GT boxes over {len(gt_by_frame)} frames for video {video_name}")
    return gt_by_frame


def evaluate(preds_by_frame, gt_by_frame, iou_thr=0.5):
    tp = fp = fn = 0
    all_frames = set(preds_by_frame.keys()) | set(gt_by_frame.keys())

    for fi in sorted(all_frames):
        preds = preds_by_frame.get(fi, [])
        gts = gt_by_frame.get(fi, [])

        used_gt = [False] * len(gts)

        for (px1, py1, px2, py2, plabel, _) in preds:
            best_iou = 0.0
            best_j = -1

            for j, (gx1, gy1, gx2, gy2, glabel) in enumerate(gts):
                if used_gt[j]:
                    continue
                if glabel != plabel:
                    continue
                v = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if v > best_iou:
                    best_iou = v
                    best_j = j

            if best_j >= 0 and best_iou >= iou_thr:
                tp += 1
                used_gt[best_j] = True
            else:
                fp += 1

        fn += sum(1 for u in used_gt if not u)

    return tp, fp, fn


def print_metrics(name, tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"=== {name} ===")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")


def summarize_rates(json_path: str, name: str):
    """
    Reads JSONL and summarizes:
      - detector call rate from inference_ran
      - propagated rate from propagated (if present)
    """
    if not os.path.exists(json_path):
        print(f"[WARN] Cannot summarize rates; file not found: {json_path}")
        return

    frames = 0
    ran = 0
    prop = 0
    prop_present = False

    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frames += 1
            if bool(rec.get("inference_ran", False)):
                ran += 1
            if "propagated" in rec:
                prop_present = True
                if bool(rec.get("propagated", False)):
                    prop += 1

    if frames == 0:
        print(f"[WARN] No frames in {json_path}")
        return

    print(f"[RATE] {name}: inference_ran {ran}/{frames} ({ran/frames:.3f})")
    if prop_present:
        print(f"[RATE] {name}: propagated {prop}/{frames} ({prop/frames:.3f})")
    print()


def evaluate_label_only(preds_by_frame, gt_by_frame, valid_classes):
    tp = fp = fn = 0
    all_frames = set(preds_by_frame.keys()) | set(gt_by_frame.keys())

    for fi in sorted(all_frames):
        preds = preds_by_frame.get(fi, [])
        gts = gt_by_frame.get(fi, [])

        pred_counts = {c: 0 for c in valid_classes}
        gt_counts = {c: 0 for c in valid_classes}

        for (_, _, _, _, plabel, _) in preds:
            if plabel in valid_classes:
                pred_counts[plabel] += 1
        for (_, _, _, _, glabel) in gts:
            if glabel in valid_classes:
                gt_counts[glabel] += 1

        for c in valid_classes:
            tpc = min(pred_counts[c], gt_counts[c])
            tp += tpc
            fp += pred_counts[c] - tpc
            fn += gt_counts[c] - tpc

    return tp, fp, fn


def greedy_iou_match(pred_boxes, gt_boxes, iou_thr):
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if v >= iou_thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_p = set()
    used_g = set()
    for _, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
    return used_p, used_g, len(used_p)


def evaluate_box_only(preds_by_frame, gt_by_frame, iou_thr=0.5):
    tp = fp = fn = 0
    all_frames = set(preds_by_frame.keys()) | set(gt_by_frame.keys())

    for fi in sorted(all_frames):
        preds = preds_by_frame.get(fi, [])
        gts = gt_by_frame.get(fi, [])

        pred_boxes = [(px1, py1, px2, py2) for (px1, py1, px2, py2, _, _) in preds]
        gt_boxes = [(gx1, gy1, gx2, gy2) for (gx1, gy1, gx2, gy2, _) in gts]

        _, _, m = greedy_iou_match(pred_boxes, gt_boxes, iou_thr)

        tp += m
        fp += len(pred_boxes) - m
        fn += len(gt_boxes) - m

    return tp, fp, fn


def sanity_counts(preds_by_frame, gt_by_frame, iou_thr):
    wrong_label_good_iou = 0
    right_label_bad_iou = 0
    total_preds = 0

    for fi in sorted(set(preds_by_frame.keys()) | set(gt_by_frame.keys())):
        preds = preds_by_frame.get(fi, [])
        gts = gt_by_frame.get(fi, [])

        gt_labels = [g[4] for g in gts]

        for (px1, py1, px2, py2, plabel, _) in preds:
            total_preds += 1

            best_any = 0.0
            best_any_label = None
            for (gx1, gy1, gx2, gy2, glabel) in gts:
                v = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if v > best_any:
                    best_any = v
                    best_any_label = glabel

            if best_any >= iou_thr and best_any_label is not None and best_any_label != plabel:
                wrong_label_good_iou += 1

            best_same = 0.0
            for (gx1, gy1, gx2, gy2, glabel) in gts:
                if glabel != plabel:
                    continue
                v = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if v > best_same:
                    best_same = v

            if (plabel in gt_labels) and (best_same < iou_thr):
                right_label_bad_iou += 1

    print(f"total_preds={total_preds}")
    print(f"wrong_label_good_iou={wrong_label_good_iou}")
    print(f"right_label_bad_iou={right_label_bad_iou}")


def run_one(name: str, preds: dict, preds_path: str, gt_by_frame: dict):
    if not preds:
        print(f"[WARN] No predictions for {name}, skipping.\n")
        return

    print(f"\n--- SANITY {name} ---")
    sanity_counts(preds, gt_by_frame, IOU_THR)

    tp, fp, fn = evaluate(preds, gt_by_frame, IOU_THR)
    print_metrics(name, tp, fp, fn)
    summarize_rates(preds_path, name)

    tp, fp, fn = evaluate_label_only(preds, gt_by_frame, VALID_CLASSES)
    print_metrics(f"{name} (label-only)", tp, fp, fn)

    tp, fp, fn = evaluate_box_only(preds, gt_by_frame, IOU_THR)
    print_metrics(f"{name} (box-only)", tp, fp, fn)
    print()


def main():
    preds_full, max_idx_full = load_predictions(PREDS_BASELINE_PATH, VALID_CLASSES)
    preds_mot, max_idx_mot = load_predictions(PREDS_SKIP_MOTION_PATH, VALID_CLASSES)
    preds_trk, max_idx_trk = load_predictions(PREDS_SKIP_MOTION_TRACK_PATH, VALID_CLASSES)

    if max_idx_full < 0 and max_idx_mot < 0 and max_idx_trk < 0:
        raise RuntimeError("No predictions loaded from any JSONL files")

    n_frames = max(max_idx_full, max_idx_mot, max_idx_trk) + 1
    print(f"VIDEO_NAME = {VIDEO_NAME}")
    print(f"N_frames (from predictions) = {n_frames}")

    gt_by_frame = load_ground_truth(GT_CSV_PATH, VIDEO_NAME, n_frames, VALID_CLASSES)

    run_one("BASELINE_FULL", preds_full, PREDS_BASELINE_PATH, gt_by_frame)
    run_one("SKIP_MOTION", preds_mot, PREDS_SKIP_MOTION_PATH, gt_by_frame)
    run_one("SKIP_MOTION_TRACK", preds_trk, PREDS_SKIP_MOTION_TRACK_PATH, gt_by_frame)


if __name__ == "__main__":
    main()
