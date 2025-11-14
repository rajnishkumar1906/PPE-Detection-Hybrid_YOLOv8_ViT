# src/extract_persons.py
import cv2
from pathlib import Path
from src.utils import ensure_dir

def extract_persons(yolo_model, input_dir, output_dir, conf=0.25):
    """
    Crop person detections from images in input_dir and save to output_dir.
    """
    ensure_dir(output_dir)
    input_dir = Path(input_dir)
    print("👤 Extracting person crops...")
    for img_path in sorted(input_dir.glob("*.*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = yolo_model.predict(source=str(img_path), conf=conf)
        if len(results) == 0 or results[0].boxes is None:
            continue
        for i, box in enumerate(results[0].boxes):
            # robust cls extraction
            try:
                cls_id = int(box.cls[0].cpu().numpy())
            except Exception:
                cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            if cls_name != "person":
                continue
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords.tolist()
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_path = Path(output_dir) / f"{img_path.stem}_person_{i}.jpg"
            cv2.imwrite(str(out_path), crop)
    print("✅ Person extraction finished.")
