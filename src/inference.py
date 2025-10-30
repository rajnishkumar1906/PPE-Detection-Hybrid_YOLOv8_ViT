# src/inference.py
from ultralytics import YOLO
import torch
import os

def run_inference(weights_path: str, test_dir: str, output_dir="runs/inference"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Running inference on {device} using model: {weights_path}")

    model = YOLO(weights_path)
    results = model.predict(source=test_dir, save=True, device=device, project=output_dir, name="predictions")

    print(f"📸 Inference done! Results saved in: {os.path.join(output_dir, 'predictions')}")
    return results
