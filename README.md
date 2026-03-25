# AssistCam: Motion-Gated Acceleration for On-Device Object Detection

Android prototype for real-time assistive scene understanding with on-device object detection and inference acceleration.
This project explores how to make mobile object detection more practical for visually impaired users by reducing unnecessary detector calls in redundant frames.

## Overview

Modern object detectors can recognize vehicles and pedestrians on mobile devices, but running a detector on every frame is computationally expensive and can reduce responsiveness.  
AssistCam investigates a lightweight video-aware acceleration strategy:

- **Baseline:** run object detection on every frame
- **SKIP_MOTION:** run detection only on motion-triggered keyframes
- **SKIP_MOTION_TRACK:** add motion-compensated box propagation between keyframes

The system is implemented on Android (Kotlin + CameraX + TensorFlow Lite) and evaluated on labeled video.

## Motivation

This project was motivated by an assistive-camera use case, where the goal is not centimeter-perfect tracking, but timely and useful warnings such as:

- "car on your left"
- "pedestrian ahead"
- "truck approaching from the right"

In that setting, approximate position and fast refresh can matter more than running a heavy detector on every single frame.

## Technical highlights

- **Platform:** Android / Kotlin
- **Camera pipeline:** CameraX (`PreviewView` + `ImageAnalysis`)
- **Model:** TensorFlow Lite EfficientDet-Lite4
- **Acceleration techniques implemented:**
  1. Motion-gated frame skipping
  2. Motion-compensated bounding box propagation
- **Evaluation pipeline:** Python script for TP / FP / FN, Precision / Recall / F1, call rate, label-only F1, and box-only F1

## Method

### 1) Baseline
Run EfficientDet-Lite4 on every frame.

### 2) SKIP_MOTION
Compute a lightweight motion score on each frame:
- downsample frame
- convert to grayscale
- compute mean absolute difference
- smooth with an exponential moving average (EMA)

Only run the detector if:
- motion exceeds a threshold, or
- the number of skipped frames exceeds a fixed limit (`maxSkipFrames`)

Otherwise, reuse the most recent detections.

### 3) SKIP_MOTION_TRACK
Extend SKIP_MOTION with a simple motion-compensation step:
- estimate approximate frame-to-frame translation
- shift previous bounding boxes during skipped frames

This is a lightweight approximation of keyframe + propagation strategies used in video detection systems.

## Key results

Evaluation video: **BDD100K** sequence `00d8944b-e157478b`  
Evaluated classes: `car`, `pedestrian`, `truck`, `bus`

### End-to-end detection results

| Mode | Precision | Recall | F1 | Detector Call Rate |
|------|-----------|--------|----|--------------------|
| BASELINE_FULL | 0.862 | 0.357 | 0.505 | 1.000 |
| SKIP_MOTION | 0.818 | 0.337 | 0.477 | 0.739 |
| SKIP_MOTION_TRACK | 0.844 | 0.352 | 0.496 | 0.877 |

### Error decomposition

| Mode | Label-only F1 | Box-only F1 |
|------|---------------|-------------|
| BASELINE_FULL | 0.558 | 0.572 |
| SKIP_MOTION | 0.550 | 0.535 |
| SKIP_MOTION_TRACK | 0.562 | 0.559 |

### Interpretation

- **SKIP_MOTION** reduces detector calls by about 26% while maintaining most of the baseline F1.
- The main degradation comes from localization drift, not class confusion.
- **SKIP_MOTION_TRACK** recovers much of the F1 loss by improving box alignment on skipped frames, though it triggers inference more often than SKIP_MOTION.

## Main Takeaways

This project shows that even a simple video-aware inference policy can recover a meaningful amount of compute on mobile object detection.

One important observation is that the performance drop under skipping is driven more by stale bounding boxes than by label errors. This motivated the second technique, motion-compensated propagation, which improved localization-related performance.

The project demonstrates a full pipeline from:
- Android deployment,
- on-device inference,
- reproducible offline benchmarking,
- to diagnostic metric design.

## Technical Context

This project combines mobile deployment, computer vision, and inference-time optimization. In addition to implementing an on-device object detector, it studies how simple video-aware scheduling strategies affect the trade-off between computational cost and detection quality.

The main technical focus is not only the detector itself, but also the surrounding inference pipeline: deciding when the detector should run, how skipped frames should be handled, and how performance should be evaluated beyond a single aggregate metric. To analyze these questions, the project includes both an Android implementation and a Python evaluation pipeline with separate label-only and box-only error analysis.

## Repository Contents

- Android Studio project for AssistCam
- Kotlin implementation of:
  - real-time camera pipeline
  - motion-gated frame skipping
  - motion-compensated propagation
- TensorFlow Lite object detection integration
- Offline evaluation mode using extracted video frames
- Python evaluation script for metrics and analysis

## Running the Project

### Android app
1. Open the project in Android Studio
2. Connect an Android device
3. Build and run the app
4. Select execution mode in `MainActivity.kt`

### Offline evaluation
The app can also run on pre-extracted image frames stored on-device and export JSONL logs.  
Those logs can be evaluated with:

```bash
python eval_assistcam.py
```

## Dataset

This project uses BDD100K for evaluation.  
BDD100K is a large-scale dataset containing diverse real-world scenes and annotations for traffic participants such as cars, trucks, buses, and pedestrians.

In this project, evaluation was performed on labeled driving video frames extracted from BDD100K. The repository does not include the full dataset, and it should be downloaded separately from the official source.

## Limitations

- The current evaluation is based on a limited number of video sequences, so the reported results should be interpreted as a prototype-level study rather than a large-scale benchmark.
- The additional tracking step improves localization in some cases, but it can also reduce the acceleration benefit if the motion gate becomes more conservative and triggers inference more often.

## Future Work

- Apply model compression or quantization to further reduce inference cost.
- Evaluate the system on a larger and more diverse set of video sequences.
- Explore stronger or slower base detectors and study whether the same video-aware acceleration strategy provides larger practical benefits in those settings.

## Methods and Tools Used

- **Android / Kotlin** for mobile application development  
- **CameraX** for real-time camera preview and frame analysis  
- **TensorFlow Lite** for on-device object detection inference  
- **EfficientDet-Lite4** as the base detection model  
- **Motion-gated frame skipping** using grayscale frame difference and exponential moving average (EMA)  
- **Motion-compensated bounding box propagation** between keyframes  
- **Python + pandas** for offline evaluation and metric analysis  
- **IoU-based matching**, along with separate label-only and box-only F1 analysis, to better understand the effects of acceleration on classification and localization
