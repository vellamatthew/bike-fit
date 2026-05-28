from ultralytics import YOLO
import numpy as np

_model = None
MODEL_PATH = "yolo11s-pose.pt"  # Small model for faster inference


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def infer(frame_bgr: np.ndarray) -> dict:
    """
    Run pose estimation on a single BGR frame.

    Returns a dict with:
        xy         : np.ndarray shape (17, 2) - x, y coordinates in pixels, or None
        xyn        : np.ndarray shape (17, 2) - normalized x, y coordinates, or None
        kpts       : np.ndarray shape (17, 3) - x, y, visibility (if available), or None
        confidence : float (person detection confidence), or None
        raw        : raw ultralytics Results object
    """
    results = get_model()(frame_bgr, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return {"xy": None, "xyn": None, "kpts": None, "confidence": None, "raw": results}

    # Take the highest-confidence detection
    if results.boxes is not None and len(results.boxes.conf) > 0:
        best_idx = int(results.boxes.conf.argmax())
    else:
        best_idx = 0

    xy = results.keypoints.xy[best_idx].cpu().numpy()        # x and y coordinates
    xyn = results.keypoints.xyn[best_idx].cpu().numpy()      # normalized
    kpts = results.keypoints.data[best_idx].cpu().numpy()    # x, y, visibility (if available)
    conf = float(results.boxes.conf[best_idx].cpu()) if results.boxes else 0.0

    return {"xy": xy, "xyn": xyn, "kpts": kpts, "confidence": conf, "raw": results}
