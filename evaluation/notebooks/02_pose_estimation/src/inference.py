# inference.py

import json
import os
import time
from pathlib import Path


# Serialized keypoint wrapper
# Reconstructs the .x / .y / .confidence interface from a loaded [x, y, conf]
# list so that evaluation code can treat it identically to live adapter output.

class SerializedKeypointWrapper:
    """Reconstructs the adapter keypoint interface from a deserialized list."""

    def __init__(self, keypoints):
        """
        Args:
            keypoints: list of [x, y, confidence] with length == total_keypoints,
                       or None if the frame had no detection.
        """
        self._keypoints = keypoints  # list of [x, y, conf]

    def __getitem__(self, idx):
        from src.model_adapters import KeypointWrapper
        kp = self._keypoints[idx]
        return KeypointWrapper(x=kp[0], y=kp[1], confidence=kp[2])


# Serialization helpers

def _extract_keypoints_from_prediction(pred, total_keypoints):
    """
    Convert a live adapter prediction (wrapper object) into a serializable
    list of [x, y, confidence] entries, one per keypoint index.

    Args:
        pred:             adapter wrapper object (supports pred[idx] -> .x .y .confidence)
        total_keypoints:  number of keypoint slots to serialize (max_index + 1)

    Returns:
        list of length total_keypoints, each entry [float, float, float]
    """
    result = []
    for idx in range(total_keypoints):
        kp = pred[idx]
        if hasattr(kp, 'confidence'):
            conf = float(kp.confidence)
        else:
            conf = 1.0  # no confidence score = always detected
        result.append([float(kp.x), float(kp.y), conf])
    return result


def save_predictions(model_name, keypoint_mapping, predictions, inference_time, output_dir):
    """
    Serialize predictions for one model to JSON.

    File is written to: {output_dir}/{model_name}.json
    Existing files are overwritten.

    Args:
        model_name:       str
        keypoint_mapping: dict {cvat_name -> keypoint_index}
        predictions:      dict {frame_num (int) -> wrapper | None}
        inference_time:   dict with keys total_seconds, per_frame_ms, fps
        output_dir:       str or Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # total_keypoints = one past the highest index used by this model
    total_keypoints = max(keypoint_mapping.values()) + 1

    serialized_predictions = {}
    for frame_num, pred in predictions.items():
        if pred is None:
            serialized_predictions[str(frame_num)] = None
        else:
            serialized_predictions[str(frame_num)] = _extract_keypoints_from_prediction(
                pred, total_keypoints
            )

    payload = {
        "model_name":      model_name,
        "keypoint_mapping": keypoint_mapping,
        "total_keypoints": total_keypoints,
        "inference_time":  inference_time,
        "predictions":     serialized_predictions,
    }

    out_path = output_dir / f"{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)

    print(f"  Saved predictions → {out_path}  "
          f"({len(serialized_predictions)} frames, "
          f"{sum(1 for v in serialized_predictions.values() if v is not None)} detections)")


def load_predictions(json_path):
    """
    Load serialized predictions from JSON.

    Returns:
        model_name:       str
        keypoint_mapping: dict {cvat_name -> int}
        predictions:      dict {frame_num (int) -> SerializedKeypointWrapper | None}
        inference_time:   dict
    """
    with open(json_path, "r") as f:
        payload = json.load(f)

    raw_preds = payload["predictions"]
    predictions = {}
    for frame_str, kps in raw_preds.items():
        frame_num = int(frame_str)
        predictions[frame_num] = SerializedKeypointWrapper(kps) if kps is not None else None

    # keypoint_mapping values come back as ints from JSON — fine.
    return (
        payload["model_name"],
        payload["keypoint_mapping"],
        predictions,
        payload["inference_time"],
    )


# Main entry points

def run_and_save_predictions(adapter, images_folder, output_dir, overwrite=False):
    """
    Run inference for one adapter and serialize to disk.
    Skips if the output file already exists (unless overwrite=True).

    Args:
        adapter:       PoseModelAdapter instance
        images_folder: path to folder of .PNG frames
        output_dir:    where to write the JSON file
        overwrite:     if False, skip models whose file already exists

    Returns:
        Path to the written (or existing) JSON file.
    """
    output_dir = Path(output_dir)
    model_name = adapter.get_model_name()
    out_path = output_dir / f"{model_name}.json"

    if out_path.exists() and not overwrite:
        print(f"  [{model_name}] Skipping — predictions already saved at {out_path}")
        return out_path

    print(f"\n{'='*70}")
    print(f"  Running inference: {model_name}")
    print(f"{'='*70}")

    predictions, total_seconds = adapter.run_inference(images_folder)

    total_frames = len(predictions)
    detections   = sum(1 for p in predictions.values() if p is not None)

    per_frame_ms = (total_seconds / total_frames * 1000) if total_frames > 0 else 0.0
    fps          = (total_frames / total_seconds)        if total_seconds > 0 else 0.0

    print(f"  Detections:  {detections}/{total_frames} ({detections/total_frames*100:.1f}%)")
    print(f"  Total time:  {total_seconds:.2f}s")
    print(f"  Per frame:   {per_frame_ms:.1f}ms")
    print(f"  FPS:         {fps:.1f}")

    inference_time = {
        "total_seconds":    total_seconds,
        "per_frame_ms":     per_frame_ms,
        "fps":              fps,
    }

    save_predictions(
        model_name       = model_name,
        keypoint_mapping = adapter.get_keypoint_mapping(),
        predictions      = predictions,
        inference_time   = inference_time,
        output_dir       = output_dir,
    )

    return out_path


def run_all_and_save(adapters, images_folder, output_dir, overwrite=False):
    """
    Convenience wrapper: run inference for a list of adapters.

    Args:
        adapters:      list of PoseModelAdapter instances
        images_folder: path to folder of .PNG frames
        output_dir:    where to write JSON files
        overwrite:     passed through to run_and_save_predictions

    Returns:
        list of output Paths (one per adapter)
    """
    paths = []
    for adapter in adapters:
        path = run_and_save_predictions(adapter, images_folder, output_dir, overwrite=overwrite)
        paths.append(path)
    return paths