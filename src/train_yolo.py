from ultralytics import YOLO
import torch

def train_yolo_model(dataset_path, device=None):
    print("🚀 Starting YOLOv8 training...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧩 Training YOLO on device: {device}")

    model = YOLO("yolov8n.pt")  # lightweight & fast
    results = model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=device,
        half=True,          # 🟢 enables mixed precision (faster on GPU)
        workers=2,
        verbose=False
    )

    weights_path = model.ckpt_path
    print("✅ YOLOv8 training done. Saved model:", weights_path)
    return weights_path
