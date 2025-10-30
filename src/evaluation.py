# src/evaluation.py

from ultralytics import YOLO
import os

def evaluate_model(weights_path: str):
    """
    Evaluates a YOLOv8 model on its validation dataset and prints key performance metrics.

    Args:
        weights_path (str): Path to YOLOv8 model weights (.pt file).

    Returns:
        results (ultralytics.yolo.engine.results.Results): YOLO evaluation results object.
    """
    print(f"📊 Evaluating YOLOv8 model: {weights_path}")

    # Check model path
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[❌] Model weights not found at {weights_path}")

    # Load the YOLO model
    model = YOLO(weights_path)

    # Run evaluation on validation data
    results = model.val()

    # Extract metrics safely
    metrics = getattr(results, "results_dict", {}) or {}
    precision = metrics.get("metrics/precision(B)", 0.0)
    recall = metrics.get("metrics/recall(B)", 0.0)
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map5095 = metrics.get("metrics/mAP50-95(B)", 0.0)

    # Print formatted metrics
    print("\n📈 Evaluation Metrics:")
    print(f"  ✅ Precision:   {precision:.4f}")
    print(f"  ✅ Recall:      {recall:.4f}")
    print(f"  ✅ mAP@50:      {map50:.4f}")
    print(f"  ✅ mAP@50-95:   {map5095:.4f}")

    print("\n💾 Evaluation complete! Results saved in the 'runs/val/' directory.")
    return results
