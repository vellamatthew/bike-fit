# Bike Fit and Posture Estimation Artefact Repository

This repository accompanies the thesis **Bike Fit and Posture Estimation using Machine Learning**. It contains the code, public data artefacts, saved predictions, evaluation outputs, trained wheel detector, and desktop demonstrator used to investigate markerless bike fitting from consumer cycling footage.

The work has two main parts:

- `evaluation/` contains the research artefacts for the pose-estimation benchmark and perspective-correction experiments.
- `app/` contains a PyQt6 desktop demonstrator that processes a cycling video and produces bike-fit measurements and recommendations.

## Repository Layout

```text
bike-fit/
  app/
    main.py                         # PyQt6 desktop application entry point
    requirements.txt                # Application dependencies
    wheel.pt                        # Trained wheel segmentation model used by the app
    inference/                      # YOLO pose inference wrapper
    processing/                     # Angle, calibration, fit assessment, correction logic
    ui/                             # PyQt6 user interface components
    workers/                        # Background processing/export workers
    bike_geometry/                  # Bike geometry lookup/storage utilities

  evaluation/
    models/
      README.md                     # Expected external third-party model layout
    notebooks/
      01_current_benchmarks/        # COCO/MPII dataset exploration
      02_pose_estimation/           # Pose benchmark data, predictions, results, analysis
      03_wheel_detection_and_perspective_correction/
                                      # Wheel segmentation and correction experiments
```

## What Is Included

The repository includes the artefacts needed to inspect the thesis results without rerunning all model inference:

- 480 annotated pose-evaluation frames and the CVAT annotation export.
- Saved predictions for the 25 pose-estimation configurations.
- Saved aggregate and per-video pose-evaluation results.
- Analysis notebooks and generated plots used in the thesis.
- Wheel-detector cross-validation metric summaries.
- `app/wheel.pt`, the trained wheel segmentation model used by the demonstrator.
- The full desktop application source code.

## What Is Not Included

Some files are intentionally omitted because they are large third-party assets or are governed by their own distribution terms:

- COCO and MPII benchmark datasets.
- The full third-party pose-model zoo used to rerun pose inference from scratch.
- Original non-anonymised Reddit source videos.
- The 165-image wheel segmentation dataset, its five-fold image/label split copies, and generated training-run visualisations.
- The 40-image perspective-correction validation set and the additional internal weight-derivation images.

See:

- [evaluation/models/README.md](evaluation/models/README.md)
- [evaluation/notebooks/01_current_benchmarks/README.md](evaluation/notebooks/01_current_benchmarks/README.md)

## Running the Desktop App

The application is documented separately in [app/README.md](app/README.md).

Quick start:

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Run the app from inside the `app/` directory. The current implementation expects `wheel.pt` to be available in the working directory, and the YOLO pose model is referenced as `yolo11s-pose.pt`.

## Evaluation Artefacts

### Pose Estimation Benchmark

The pose-estimation evaluation is in:

```text
evaluation/notebooks/02_pose_estimation/
```

Important contents:

- `data/annotations/annotations.xml` is the CVAT annotation export.
- `data/annotations/images/default/` contains the 480 face-blurred evaluation frames.
- `predictions/` contains saved predictions for all evaluated model configurations.
- `results/` contains full and aggregated evaluation metrics.
- `plots/` contains figures used in the thesis.
- `src/` contains reusable loading, inference, metrics, and evaluation code.

Notebook order:

1. `01_blur_faces.ipynb` documents the face-blurring step.
2. `02_evaluate_models.ipynb` evaluates predictions and can rerun inference if external models are supplied.
3. `03_analysis.ipynb` produces the comparison tables and figures used in the thesis.

For result inspection, start with `03_analysis.ipynb`. To rerun pose inference from scratch, place the omitted model files under `evaluation/models/` as described in [evaluation/models/README.md](evaluation/models/README.md).

### Existing Benchmark Dataset Analysis

The notebooks in:

```text
evaluation/notebooks/01_current_benchmarks/
```

inspect how COCO and MPII represent cycling-relevant poses. The datasets themselves are not included. Their expected layout is documented in that folder's README.

### Wheel Segmentation and Perspective Correction

The perspective-correction experiments are in:

```text
evaluation/notebooks/03_wheel_detection_and_perspective_correction/
```

Important contents:

- `01_model_training.ipynb` builds five-fold splits, trains the YOLO wheel segmentation model, validates each fold, and trains the final deployment model.
- `02_perspective_correction.ipynb` evaluates the ellipse-based homography correction method.
- The annotated wheel segmentation dataset and perspective-correction image sets are not redistributed in this public artefact repository.
- `data/kfold_5_splits/kfold_metrics_summary.csv` contains fold-level segmentation metrics.
- Generated training-run visualisations derived from the internal image sets are omitted.
- `plots/fig_loss_discrimination.pdf` and `plots/fig_correction_pairs.pdf` correspond to the thesis figures.

The final wheel model is copied into `app/wheel.pt` for use by the demonstrator.

## Key Results Stored in This Repository

The saved artefacts reproduce the thesis-level findings:

- ViTPose++ Large trained on MPII achieved the lowest mean joint-angle RMSE among evaluated pose models.
- MPII-trained ViTPose++ variants achieved lower angle error than COCO-trained equivalents, despite COCO variants having slightly higher keypoint-localisation AUC.
- OpenPose MPI achieved strong angle accuracy but lower lower-limb detection rates, making detection reliability important for deployment.
- The wheel segmentation model achieved mean segmentation mAP@0.5:0.95 of approximately `0.946 +/- 0.043` across five folds.
- The perspective-correction notebook demonstrates automatic camera-alignment correction from fitted wheel ellipses.

## Ethics Statement

The cycling footage was sourced from public r/bikefit posts as described in the thesis. The repository contains face-blurred frames and annotation artefacts, not the original non-anonymised videos. No Reddit usernames or post metadata are included.

## Reproducibility Notes

The saved predictions and result JSON files are the most practical way to reproduce the thesis analysis without reconstructing every external model environment. Full inference reruns require a heavier setup because the evaluated families use different frameworks and model formats, including Ultralytics YOLO, MediaPipe, HRNet, OpenPose, Lightweight OpenPose, and Hugging Face ViTPose models.

The desktop app dependencies are listed separately in `app/requirements.txt`.
