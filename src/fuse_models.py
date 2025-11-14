# src/fuse_models.py
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.train_vit import PPE_CLASSES

def fuse_models(yolo_model, vit_model, image_path, conf_threshold=0.25):
    """
    Run YOLO (person detection), crop persons, apply ViT per crop to detect PPE.
    Returns structured dict with persons list and overall score.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = yolo_model.predict(source=image_path, conf=conf_threshold)
    persons = []
    vit_model.eval()

    if len(results) == 0:
        return {"persons": [], "overall_score": 0.0, "status": "No detections"}

    for box in results[0].boxes:
        try:
            cls_id = int(box.cls[0].cpu().numpy())
        except Exception:
            cls_id = int(box.cls[0])
        cls_name = yolo_model.names.get(cls_id, None)
        if cls_name != "person":
            continue

        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords.tolist()
        h, w = img.shape[:2]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_t = vit_transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = vit_model(crop_t)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        ppe_dict = {PPE_CLASSES[i]: bool(probs[i] > 0.5) for i in range(len(PPE_CLASSES))}
        score = float(sum(ppe_dict.values()) / len(PPE_CLASSES))
        persons.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "ppe": ppe_dict,
            "compliance_score": score
        })

    if not persons:
        return {"persons": [], "overall_score": 0.0, "status": "No persons detected"}

    overall = float(np.mean([p["compliance_score"] for p in persons]))
    if overall >= 0.8:
        status = "Compliant"
    elif overall >= 0.5:
        status = "Partially Compliant"
    else:
        status = "Non-Compliant"

    return {"persons": persons, "overall_score": overall, "status": status}
