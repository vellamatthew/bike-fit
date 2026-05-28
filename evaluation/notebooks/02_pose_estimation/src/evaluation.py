# evaluation.py

import json
import numpy as np
from pathlib import Path

from .metrics import evaluate_model
from .inference import load_predictions


# Run evaluation over all saved prediction files

def run_evaluation(predictions_dir, videos):
    """
    Load all serialized prediction files and evaluate each model.

    Args:
        predictions_dir: path to folder containing {model_name}.json files
        videos:          dict from split_frames_into_videos()

    Returns:
        dict {model_name -> {
            'aggregated':       aggregated metrics,
            'per_video':        per-video metrics,
            'inference_time':   timing dict from inference,
            'keypoint_mapping': keypoint mapping used,
        }}
    """
    predictions_dir = Path(predictions_dir)
    json_files = sorted(predictions_dir.glob('*.json'))

    if not json_files:
        print(f"No prediction files found in {predictions_dir}")
        return {}

    all_results = {}

    for json_path in json_files:
        model_name, keypoint_mapping, predictions, inference_time = load_predictions(json_path)

        print(f"\n{'='*70}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*70}")

        results = evaluate_model(videos, predictions, keypoint_mapping)

        agg = results['aggregated']
        print(f"  PCKh@0.5:        {agg['pckh'][0.5]['mean']:.1f}% "
              f"(±{agg['pckh'][0.5]['std']:.1f}%)")
        print(f"  AUC:             {agg['auc']['overall']['mean']:.1f}% "
              f"(±{agg['auc']['overall']['std']:.1f}%)")
        print(f"  Detection Rate:  {agg['detection_rate']['overall']['mean']*100:.1f}% "
              f"(±{agg['detection_rate']['overall']['std']*100:.1f}%)")

        angle_overall = agg['angle_mae']['overall']
        if angle_overall['mean'] is not None:
            per_joint_str = '  '.join(
                f"{name}: {vals['mean']:.1f}°"
                for name, vals in sorted(agg['angle_mae']['per_joint'].items())
                if vals['mean'] is not None
            )
            print(f"  Angle MAE:       {angle_overall['mean']:.1f}° "
                  f"(±{angle_overall['std']:.1f}°)  [{per_joint_str}]")
        
        angle_rmse_overall = agg['angle_rmse']['overall']
        angle_me_overall   = agg['angle_me']['overall']
        if angle_rmse_overall['mean'] is not None:
            print(f"  Angle RMSE:      {angle_rmse_overall['mean']:.1f}°")
            print(f"  Angle ME:        {angle_me_overall['mean']:.1f}°")


        print(f"  FPS:             {inference_time['fps']:.1f}")

        all_results[model_name] = {
            'aggregated':       agg,
            'per_video':        results['per_video'],
            'inference_time':   inference_time,
            'keypoint_mapping': keypoint_mapping,
        }

    return all_results


# Save / load results

def save_results(all_results, output_dir):
    """
    Save evaluation results for all models to JSON.

    Writes two files per model:
        {model_name}_aggregated.json  — aggregated metrics only (human-readable)
        {model_name}_full.json        — aggregated + per_video + timing + mapping

    Args:
        all_results: dict as returned by run_evaluation()
        output_dir:  str or Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, results in all_results.items():
        agg_path = output_dir / f"{model_name}_aggregated.json"
        with open(agg_path, 'w') as f:
            json.dump(_to_serializable(results['aggregated']), f, indent=2)

        full_path = output_dir / f"{model_name}_full.json"
        with open(full_path, 'w') as f:
            json.dump(_to_serializable(results), f, indent=2)

        print(f"  Saved {model_name} -> {output_dir}")


def load_results(model_name, results_dir):
    """
    Load full saved results for one model.

    Note: JSON keys that were floats (PCKh thresholds) come back as strings.
    Use results['aggregated']['pckh']['0.5'] not ['pckh'][0.5] after loading.

    Args:
        model_name:  str
        results_dir: str or Path

    Returns:
        dict with keys: aggregated, per_video, inference_time, keypoint_mapping
    """
    path = Path(results_dir) / f"{model_name}_full.json"
    with open(path, 'r') as f:
        return json.load(f)


def load_all_results(results_dir):
    """
    Load all saved full results from a directory.

    Returns:
        dict {model_name -> results_dict}
    """
    results_dir = Path(results_dir)
    all_results = {}

    for path in sorted(results_dir.glob('*_full.json')):
        model_name = path.stem.replace('_full', '')
        with open(path, 'r') as f:
            all_results[model_name] = json.load(f)

    return all_results


# Serialization helper

def _to_serializable(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj