# Current Benchmark Dataset Analysis

This folder contains exploratory notebooks used to inspect existing human pose estimation benchmark datasets in relation to cycling and bike fitting.

The benchmark datasets themselves are not included in this repository because they are large third-party datasets with their own distribution terms.

## Notebooks

- `COCO_analysis.ipynb` analyses the COCO keypoint dataset.
- `MPII_analysis.ipynb` analyses the MPII Human Pose dataset.

These notebooks were used to understand how existing benchmark datasets represent cycling-relevant poses and keypoints.

## Expected Data Structure

To rerun the notebooks, download the datasets from their official sources and place them in the following structure:

```text
notebooks/01_current_benchmarks/data/
  COCO/
    annotations/
      person_keypoints_train2017.json
      person_keypoints_val2017.json
      instances_train2017.json
      instances_val2017.json
    train2017/
      train2017/
        *.jpg
    val2017/
      val2017/
        *.jpg

  MPII/
    mpii_human_pose_v1_u12_1.mat
    images/
      *.jpg
```

## Dataset Sources

- COCO: https://cocodataset.org/
- MPII Human Pose Dataset: http://human-pose.mpi-inf.mpg.de/