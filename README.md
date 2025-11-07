
# 🦺 PPE Compliance Detection System

A *robust, AI-powered system* for monitoring Personal Protective Equipment (PPE) compliance in workplaces.  
The system uses *YOLOv8* for person detection and *Vision Transformer (ViT)* for PPE classification on cropped person images. It also provides a *web interface* for uploading images and visualizing results in real-time.

---

## 🔍 Features

- Detects persons in images and identifies their PPE compliance.
- Supports detection of multiple PPE items: boots, ear protection, goggles, gloves, helmet, mask, vest.
- Provides *real-time compliance scores* for each detected person.
- Uses *YOLOv8* for object detection and *ViT* for classification on cropped persons.
- Handles *noisy images* with preprocessing to improve accuracy.
- *Flask-based web app* for easy image upload and results visualization.

---

## 🗂 Project Structure

object_detection/ ├── app.py                  # Flask app entry point ├── src/                    # All source code modules │   ├── train_yolo.py       # YOLOv8 training script │   ├── train_vit.py        # ViT training script │   ├── extract_persons.py  # Crop persons from images │   ├── auto_label_ann.py   # Auto-label PPE on cropped persons │   ├── fuse_models.py      # Fuse YOLO + ViT predictions │   └── utils.py            # Utility functions ├── dataset/                # Dataset folder (images & YAML) │   ├── train/ │   ├── valid/ │   └── test/ ├── weights/                # Saved YOLO and ViT weights ├── tmp/                    # Temporary folder: person crops & annotations │   ├── persons/ │   └── ann.json ├── static/                 # Web app static folder │   ├── uploads/ │   └── results/ └── README.md               # Project documentation

---

## ⚙ Installation

1. Clone the repository:

```bash
git clone https://github.com/username/ppe-compliance-detection.git
cd ppe-compliance-detection

2. Create a virtual environment (Python 3.12 recommended):



conda create -n ppe python=3.12 -y
conda activate ppe

3. Install required packages:



pip install -r requirements.txt

4. Prepare your dataset:



dataset/
├── train/images/
├── valid/images/
└── test/images/

5. Ensure data.yaml exists with class names:



nc: 8
names: ['boots', 'ear_protection', 'goggles', 'gloves', 'helmet', 'mask', 'person', 'vest']


---

🚀 Usage

1. Extract Person Crops

python src/extract_persons.py

Crops all detected persons into tmp/persons/ folder.


---

2. Auto-label PPE (Optional)

Automatically label PPE on cropped images:

python src/auto_label_ann.py

> This generates tmp/ann.json with PPE labels.




---

3. Train Models (if not already trained)

YOLOv8 will detect persons and PPE objects.

ViT will classify PPE compliance on cropped person images.


> The Flask app automatically trains ViT if vit_best.pt is missing and person crops & annotations are present.




---

4. Run Web App

python app.py

Open your browser: http://127.0.0.1:5000

Upload an image and view compliance results.

Compliance is visualized using colored bounding boxes:

Green: ≥80% PPE compliance

Orange: 50–79% compliance

Red: <50% compliance




---

🎯 Key Features in Action

Fusion of YOLO and ViT predictions.

Compliance scoring based on PPE detection.

Preprocessing and noise removal to improve prediction accuracy.

Support for multiple PPE classes.

Real-time web interface for user-friendly interaction.



---

💻 Requirements

Python 3.12

PyTorch

timm for Vision Transformer

ultralytics for YOLOv8

OpenCV

Flask


Example requirements.txt:

torch>=2.1.0
timm>=0.9.0
ultralytics>=8.0.20
opencv-python>=4.8.0
Flask>=2.3.0
numpy>=1.25.0
Pillow>=10.0
tqdm>=4.65


---

📝 Notes

Ensure person crops exist in tmp/persons/ before training ViT.

Fill ann.json either manually or with auto_label_ann.py.

YOLOv8 weights should be saved in weights/yolo_best.pt.

Training ViT for more epochs increases accuracy (default 15, can be changed in app.py).



---

📦 Future Improvements

Add real-time video streaming for live PPE monitoring.

Integrate alert system for non-compliant workers.

Support more PPE classes or custom datasets.



---

⚡ Authors

Rajnish – Original Developer – Your GitHub Profile


---

🛡 License

MIT License – see LICENSE file.


---

This README is *ready-to-go*.  

It clearly explains the *project, usage, structure, and how to get it running, making it **GitHub-ready*.  

---

If you want, I can also **create a ready-to-copy requirements.txt** and *push commands* snippet so anyone can clone and run the project immediately.  

Do you want me to do that too?
