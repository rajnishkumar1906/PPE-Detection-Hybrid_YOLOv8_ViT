# src/train_yolo.py
import os
from ultralytics import YOLO
from src.utils import ensure_dir

def train_yolo(data_yaml, project_dir, epochs=50, imgsz=640, batch=16):
    """
    Train YOLOv8s using ultralytics API.
    Returns (model, best_path).
    """
    ensure_dir(project_dir)
    abs_yaml = os.path.abspath(data_yaml)
    if not os.path.exists(abs_yaml):
        raise FileNotFoundError(f"Dataset YAML not found at: {abs_yaml}")

    print(f"🚀 Training YOLOv8s with dataset: {abs_yaml}")
    model = YOLO("yolov8n.pt")  # use yolov8s base

    batch = min(batch, 8)
    try:
        model.train(
            data=abs_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project_dir,
            name="ppe_yolo",
            exist_ok=True,
            workers=0
        )
    except Exception as e:
        raise RuntimeError(f"❌ YOLO training failed: {e}")

    best = os.path.join(project_dir, "ppe_yolo", "weights", "best.pt")
    if not os.path.exists(best):
        alt = os.path.join(project_dir, "ppe_yolo", "weights", "last.pt")
        if os.path.exists(alt):
            best = alt
        else:
            raise FileNotFoundError("⚠️ Training finished but best.pt not found — check YOLO logs.")
    print(f"✅ YOLO training complete. Model saved at: {best}")
    return model, best
