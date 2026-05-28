# External Model Files

The full third-party model zoo is not included in this repository because the files are large and are distributed by their original authors.

The pose-estimation analysis can be reproduced from the saved prediction files in:

- `notebooks/02_pose_estimation/predictions/`
- `notebooks/02_pose_estimation/results/`

The model files are only required if you want to rerun pose inference from scratch using `notebooks/02_pose_estimation/02_evaluate_models.ipynb`.

## Expected Structure

Place external model files in the following structure:

```text
models/
  yolo/
    yolo11n-pose.pt
    yolo11s-pose.pt
    yolo11m-pose.pt
    yolo11l-pose.pt
    yolo11x-pose.pt
    yolo26n-seg.pt

  mediapipe/
    pose_landmarker_lite.task
    pose_landmarker_full.task
    pose_landmarker_heavy.task

  hrnet/
    models/pytorch/pose_coco/
      pose_hrnet_w32_256x192.pth
      pose_hrnet_w32_384x288.pth
      pose_hrnet_w48_384x288.pth

  lightweight_openpose/
    checkpoint_iter_370000.pth

  openpose/
    bin/OpenPoseDemo.exe
    models/pose/body_25/pose_iter_584000.caffemodel
    models/pose/coco/pose_iter_440000.caffemodel
    models/pose/mpi/pose_iter_160000.caffemodel