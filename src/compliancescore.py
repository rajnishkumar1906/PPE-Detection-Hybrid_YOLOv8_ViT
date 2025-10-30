# src/compliancescore.py

import os
import glob
import json

def calculate_compliance_score(pred_dir="runs/inference/predictions"):
    """
    Calculates PPE compliance score based on prediction results.
    Works with image filenames OR YOLO JSON exports.
    """

    print("🦺 Calculating PPE compliance score...")

    if not os.path.exists(pred_dir):
        print(f"[❌] Predictions directory not found: {pred_dir}")
        return None

    # Gather prediction files
    img_files = glob.glob(os.path.join(pred_dir, "*.jpg")) + glob.glob(os.path.join(pred_dir, "*.png"))
    json_files = glob.glob(os.path.join(pred_dir, "*.json"))

    if not img_files and not json_files:
        print("[⚠️] No prediction files found in directory.")
        return None

    total = 0
    compliant = 0

    # --- Case 1: From image filenames ---
    if img_files:
        for file in img_files:
            total += 1
            name = os.path.basename(file).lower()

            # If no 'nohelmet', 'nogloves', etc. detected in name, mark compliant
            if all(x not in name for x in ["nohelmet", "nogloves", "novest", "nomask"]):
                compliant += 1

    # --- Case 2: From JSON (YOLOv8 detection exports) ---
    elif json_files:
        for file in json_files:
            total += 1
            with open(file, "r") as f:
                data = json.load(f)
            detections = [obj["name"].lower() for obj in data.get("objects", [])]

            # Check if any "No" PPE detections exist
            if all(not det.startswith("no") for det in detections):
                compliant += 1

    # --- Final compliance calculation ---
    score = (compliant / total) * 100 if total > 0 else 0.0

    print("\n📊 PPE Compliance Summary:")
    print(f"   Total Images Analyzed : {total}")
    print(f"   Fully Compliant Images: {compliant}")
    print(f"   Non-Compliant Images  : {total - compliant}")
    print(f"   ✅ Compliance Score    : {score:.2f}%")

    return {
        "total_images": total,
        "compliant": compliant,
        "non_compliant": total - compliant,
        "compliance_score": round(score, 2)
    }


if __name__ == "__main__":
    calculate_compliance_score()
