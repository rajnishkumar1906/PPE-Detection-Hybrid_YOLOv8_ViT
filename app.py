# app.py
import os
import json
import multiprocessing
import cv2
import torch
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import timm

# local imports
from src.utils import ensure_dir, get_device
from src.train_yolo import train_yolo
from src.extract_persons import extract_persons
from src.train_vit import train_vit, PPE_CLASSES
from src.fuse_models import fuse_models
from src.auto_label_ann import auto_label_ann  # new integrated auto-label

# -------- CONFIG ----------
BASE = os.path.abspath(r"D:\COMPUTERVISION\object_detection")  # change if needed
DATA_YAML = os.path.join(BASE, "dataset", "data.yaml")
TMP = os.path.join(BASE, "tmp")
PERSON_DIR = os.path.join(TMP, "persons")
ANN_FILE = os.path.join(TMP, "ann.json")
WEIGHTS = os.path.join(BASE, "weights")

ensure_dir(TMP)
ensure_dir(PERSON_DIR)
ensure_dir(WEIGHTS)
ensure_dir(os.path.join(BASE, "static", "uploads"))
ensure_dir(os.path.join(BASE, "static", "results"))

device = get_device()

# flask
app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = os.path.join(BASE, "static", "uploads")
app.config["RESULT_FOLDER"] = os.path.join(BASE, "static", "results")

# global models
yolo_model = None
vit_model = None

# Hyperparams
VIT_EPOCHS = 20   # change if you want more
YOLO_EPOCHS = 50  # leave as-is or change

# image enhancement (simple denoise + CLAHE + sharpen)
def enhance_image_path(path):
    img = cv2.imread(path)
    if img is None:
        return
    # denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # sharpen (unsharp mask)
    gauss = cv2.GaussianBlur(img, (0,0), 3)
    img = cv2.addWeighted(img, 1.5, gauss, -0.5, 0)
    cv2.imwrite(path, img)

def prepare_models(skip_train_if_weights_exist=True, force_auto_label=False):
    """
    Loads (or trains) YOLO and ViT. Also runs auto-labeling (if needed)
    so ann.json will be created from person crops using a PPE detector.
    """
    global yolo_model, vit_model

    # --- YOLO (prefer user-trained weight in weights/yolo_best.pt) ---
    user_yolo = os.path.join(WEIGHTS, "yolo_best.pt")
    if skip_train_if_weights_exist and os.path.exists(user_yolo):
        print("✅ Loading YOLO from weights/yolo_best.pt")
        yolo_model = YOLO(user_yolo)
    else:
        print("🔧 No user YOLO weights found — training YOLO (yolov8s) ...")
        # train_yolo returns (model, best_path)
        yolo_model, best = train_yolo(DATA_YAML, BASE, epochs=YOLO_EPOCHS)
        try:
            import shutil
            shutil.copy(best, user_yolo)
        except Exception:
            pass

    # --- ensure person crops exist (extract only if missing) ---
    if len(os.listdir(PERSON_DIR)) == 0:
        print("👤 No person crops found — extracting from train images ...")
        train_images_dir = os.path.join(BASE, "dataset", "train", "images")
        extract_persons(yolo_model, train_images_dir, PERSON_DIR, conf=0.25)

    # --- AUTO-LABEL ann.json if missing or forced ---
    if force_auto_label or not os.path.exists(ANN_FILE) or _ann_empty(ANN_FILE):
        print("🧠 Auto-labeling person crops to create ann.json ...")
        # auto_label_ann will try: weights/yolo_best.pt -> yolov8s.pt fallback
        auto_label_ann(yolo_model=yolo_model, person_dir=PERSON_DIR, out_ann=ANN_FILE)

    # --- ViT (train if missing) ---
    vit_path = os.path.join(WEIGHTS, "vit_best.pt")
    if skip_train_if_weights_exist and os.path.exists(vit_path):
        print("✅ Loading ViT from weights/vit_best.pt")
        vit_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(PPE_CLASSES))
        vit_model.load_state_dict(torch.load(vit_path, map_location=device))
        vit_model.to(device).eval()
    else:
        print("🔧 Training ViT (using person crops + ann.json)...")
        vit_model = train_vit(PERSON_DIR, ANN_FILE, WEIGHTS, epochs=VIT_EPOCHS)

    return yolo_model, vit_model

def _ann_empty(path):
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
            # empty or no ppe labels anywhere
            if not data:
                return True
            for v in data.values():
                if v.get("ppe_labels"):
                    return False
            return True
    except Exception:
        return True

@app.route("/", methods=["GET", "POST"])
def index():
    global yolo_model, vit_model
    if yolo_model is None or vit_model is None:
        yolo_model, vit_model = prepare_models(skip_train_if_weights_exist=True, force_auto_label=False)

    uploaded_image = None
    result_image = None
    result = None
    error = None

    if request.method == "POST":
        f = request.files.get("file")
        if not f:
            return redirect(request.url)
        filename = f.filename
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(save_path)

        # enhance before predict (optional - improves detection sometimes)
        enhance_image_path(save_path)
        uploaded_image = url_for("static", filename=f"uploads/" + filename)

        try:
            out = fuse_models(yolo_model, vit_model, save_path, conf_threshold=0.25)
            result = out

            # draw boxes and save result image
            img = cv2.imread(save_path)
            for p in out.get("persons", []):
                x1, y1, x2, y2 = p["bbox"]
                score = p.get("compliance_score", 0)
                color = (0,255,0) if score>=0.8 else (0,165,255) if score>=0.5 else (0,0,255)
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                label = f"{int(score*100)}%"
                cv2.putText(img, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            result_fname = f"result_{filename}"
            result_path = os.path.join(app.config["RESULT_FOLDER"], result_fname)
            cv2.imwrite(result_path, img)
            result_image = url_for("static", filename=f"results/{result_fname}")

        except Exception as e:
            error = str(e)
            print("❌ Error processing:", e)

    return render_template("index.html",
                           uploaded_image=uploaded_image,
                           result_image=result_image,
                           result=result,
                           error=error)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("🌍 Starting app ...")
    # prepare models up-front so first request is fast (optional)
    yolo_model, vit_model = prepare_models(skip_train_if_weights_exist=True)
    app.run(debug=False, host="0.0.0.0", port=5000)
