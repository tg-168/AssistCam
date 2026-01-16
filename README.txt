Software Requirement: Android Studio, Python
Hardware Requirement: Android Phone

Live Demo steps:
Open the Android Studio project.
Connect an Android phone (USB debugging enabled).
In MainActivity.kt, call startCamera() in onCreate() (instead of runOfflineEval).
Select mode in MainActivity.kt:
	BASELINE_FULL
	SKIP_MOTION
	SKIP_MOTION_TRACK
	
Run the app. Bounding boxes and stats (FPS / calls per sec / latency) will appear on screen.

Offline Evaluation steps:
Open the Android Studio project.
Connect an Android phone (USB debugging enabled).
Push frames into:
	/Android/data/edu.css490.assistcam2/files/eval_frames/seq01/

In MainActivity.kt, call:
	runOfflineEval(Mode.BASELINE_FULL)
	runOfflineEval(Mode.SKIP_MOTION)
	runOfflineEval(Mode.SKIP_MOTION_TRACK)

Run the app. It will process all frames and write JSONL logs to:
	/Android/data/edu.css490.assistcam2/files/eval_MODE.jsonl

Put files in the same folder:
	mot_labels.csv (ground truth)
	eval_BASELINE_FULL.jsonl
	eval_SKIP_MOTION.jsonl
	eval_SKIP_MOTION_TRACK.jsonl
	eval_assistcam.py

Run python script