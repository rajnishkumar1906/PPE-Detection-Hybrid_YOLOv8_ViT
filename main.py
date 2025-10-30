# main.py

import argparse
import torch
import os

from src.train_yolo import train_yolo_model
from src.train_vit import train_vit_model
from src.hybrid_fusion import fuse_models
from src.inference import run_inference
from src.evaluation import evaluate_model
from src.compliancescore import calculate_compliance_score

def log_device():
    """Display current device info."""
    if torch.cuda.is_available():
        print(f"🟢 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available — using CPU")

def safe_run(step_name, func, *args):
    """Run each pipeline step safely and handle errors gracefully."""
    try:
        print(f"\n{'='*60}\n▶️ Starting step: {step_name}\n{'='*60}")
        result = func(*args)
        print(f"✅ Completed: {step_name}\n")
        return result
    except Exception as e:
        print(f"❌ Error in {step_name}: {e}\n")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid PPE Detection using YOLOv8 + ViT")

    parser.add_argument("--train-yolo", action="store_true", help="Train YOLOv8 model")
    parser.add_argument("--train-vit", action="store_true", help="Train ViT model")
    parser.add_argument("--fuse", action="store_true", help="Fuse YOLO and ViT features")
    parser.add_argument("--inference", action="store_true", help="Run inference on test images")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate YOLO model performance")
    parser.add_argument("--compliance", action="store_true", help="Calculate compliance score")
    parser.add_argument("--run-all", action="store_true", help="Run the complete pipeline")

    parser.add_argument("--dataset-path", type=str, default="dataset", help="Dataset root folder")
    parser.add_argument("--yolo-weights", type=str, default="runs/ppe_yolo_train/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--vit-weights", type=str, default="runs/vit/best_vit.pth", help="Path to ViT weights")
    parser.add_argument("--test-dir", type=str, default="dataset/test/images", help="Path to test images")

    args = parser.parse_args()

    os.makedirs("runs", exist_ok=True)
    log_device()

    # Run the full pipeline
    if args.run_all:
        print("\n🚀 Starting Full Hybrid PPE Detection Pipeline...\n")

        safe_run("YOLOv8 Training", train_yolo_model, args.dataset_path)
        safe_run("Vision Transformer Training", train_vit_model, args.dataset_path)
        safe_run("Hybrid Model Fusion", fuse_models, args.yolo_weights, args.vit_weights)
        safe_run("YOLOv8 Evaluation", evaluate_model, args.yolo_weights)
        safe_run("Inference on Test Set", run_inference, args.yolo_weights, args.test_dir)
        safe_run("Compliance Score Calculation", calculate_compliance_score)

        print("\n🎯 Full Pipeline Execution Completed Successfully!")

    else:
        # Run individual stages
        if args.train_yolo:
            safe_run("YOLOv8 Training", train_yolo_model, args.dataset_path)
        if args.train_vit:
            safe_run("Vision Transformer Training", train_vit_model, args.dataset_path)
        if args.fuse:
            safe_run("Hybrid Model Fusion", fuse_models, args.yolo_weights, args.vit_weights)
        if args.evaluate:
            safe_run("YOLOv8 Evaluation", evaluate_model, args.yolo_weights)
        if args.inference:
            safe_run("Inference on Test Set", run_inference, args.yolo_weights, args.test_dir)
        if args.compliance:
            safe_run("Compliance Score Calculation", calculate_compliance_score)
