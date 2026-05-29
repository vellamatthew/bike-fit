# Bike Fit Analyser App

This folder contains the PyQt6 desktop demonstrator developed for the thesis **Bike Fit and Posture Estimation using Machine Learning**.

The app processes a side-view cycling video, estimates rider pose using YOLO11, extracts joint angles across the pedal stroke, identifies critical pedal positions, and compares the resulting measurements against discipline-specific target ranges. It can optionally apply the wheel-based perspective-correction method developed in the thesis.

The trained wheel model used by the demonstrator is included as `wheel.pt`. The internal wheel-segmentation and perspective-correction image datasets used during development are not redistributed with this public repository.

## Features

- Drag-and-drop or file-picker video loading.
- Frame-by-frame YOLO11 pose estimation.
- Automatic or manual side selection.
- Knee, hip, elbow, ankle, and back-angle measurement logic.
- Pedal-stroke analysis with top-dead-centre and bottom-dead-centre detection.
- Discipline-specific fit assessment for road, mountain, time-trial, and triathlon positions.
- Optional Savitzky-Golay smoothing of angle waveforms.
- Optional perspective correction using detected bicycle wheels.
- Optional per-frame perspective correction.
- Optional wheelbase-based calibration for approximate body measurements in millimetres.
- Annotated video export and in-app playback.

## Folder Structure

```text
app/
  main.py                  # Application entry point
  requirements.txt         # Python dependencies
  wheel.pt                 # Trained YOLO wheel segmentation model

  inference/
    yolo_pose.py           # YOLO11 pose model wrapper

  processing/
    angles.py              # Joint-angle calculations
    critical_positions.py  # TDC/BDC extraction
    fit_assessment.py      # Target ranges and recommendations
    perspective_correction.py
                             # Wheel detection, ellipse fitting, homography correction
    calibration.py         # Wheelbase-based scale calibration
    annotate.py            # Skeleton drawing
    angle_overlay.py       # Export overlay rendering

  ui/                      # PyQt6 widgets and dialogs
  workers/                 # Background video processing/export workers
  bike_geometry/           # Bike geometry lookup and local storage
  output_videos/           # Generated annotated videos, when present
```

## Setup

From the repository root:

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU acceleration, install a PyTorch build that matches your CUDA setup before running the app.

## Running

Run the app from inside this folder:

```powershell
python main.py
```

This working directory matters because the current implementation looks for `wheel.pt` in the current directory.

The app's pose wrapper references:

```text
yolo11s-pose.pt
```

Ultralytics can normally resolve/download this model by name. If running offline, place `yolo11s-pose.pt` in this folder or adjust `MODEL_PATH` in `inference/yolo_pose.py`.

## Basic Workflow

1. Launch the app with `python main.py`.
2. Load a side-view cycling video using the drop zone or file picker.
3. Choose the cycling discipline if needed.
4. Optionally enable perspective correction before processing.
5. Click **Process Video**.
6. Confirm wheel detections if perspective correction is enabled.
7. Review the angle plots, representative frames, fit summary, and recommendations.
8. Use the exported annotated video if visual inspection is needed.

## Troubleshooting

If `wheel.pt` is not found, make sure the app is being launched from the `app/` directory.

If `yolo11s-pose.pt` cannot be loaded, run once with internet access so Ultralytics can resolve the model, or place the file locally and update `MODEL_PATH`.

If video processing is very slow, use a shorter clip, accept the long-video trim prompt, or run with a GPU-enabled PyTorch installation.

If perspective correction fails, rerun without it or use footage where both wheels are fully visible.
