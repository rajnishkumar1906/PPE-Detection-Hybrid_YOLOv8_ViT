# src/auto_label_ann.py
import os, json
from tqdm import tqdm
from ultralytics import YOLO

PPE_ITEMS = ["helmet", "vest", "gloves", "mask"]

def auto_label_ann(yolo_model=None, person_dir="tmp/persons", out_ann="tmp/ann.json", conf=0.25):
    """
    Auto-label person crops for PPE by running a PPE-capable YOLO model on each crop.
    Tries the provided yolo_model first. If not provided, attempts weights/yolo_best.pt,
    otherwise falls back to public yolov8s.pt model (pretrained).
    Saves ann.json mapping filename -> {"ppe_labels": [...]}.
    """
    # determine model to use
    model = None
    if yolo_model is not None:
        model = yolo_model
    else:
        # try user trained weight first
        if os.path.exists("weights/yolo_best.pt"):
            model = YOLO("weights/yolo_best.pt")
        else:
            # fallback to public YOLOv8s checkpoint (detects COCO classes only; may miss PPE)
            model = YOLO("yolov8s.pt")

    files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    annotations = {}
    print(f"🧠 Auto-labeling {len(files)} person crops using model -> {model.model.yaml.get('name','YOLO') if hasattr(model,'model') else 'yolov8s'}")
    for f in tqdm(files):
        path = os.path.join(person_dir, f)
        try:
            res = model.predict(source=path, conf=conf)
            detected = set()
            if len(res) > 0 and res[0].boxes is not None:
                for box in res[0].boxes:
                    # robustly get class name
                    try:
                        cls_id = int(box.cls[0].cpu().numpy())
                    except Exception:
                        cls_id = int(box.cls[0])
                    name = model.names.get(cls_id, None)
                    if name in PPE_ITEMS:
                        detected.add(name)
            annotations[f] = {"ppe_labels": sorted(list(detected))}
        except Exception:
            annotations[f] = {"ppe_labels": []}

    # save
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)
    with open(out_ann, "w") as fh:
        json.dump(annotations, fh, indent=2)
    print(f"✅ Auto-labeling complete. Saved annotations at: {out_ann}")
    # show example
    example = next(iter(annotations.items())) if annotations else None
    print("🧾 Example entry:", example)
    return out_ann
