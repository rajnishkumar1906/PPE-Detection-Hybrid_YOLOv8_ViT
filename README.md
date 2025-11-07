

PPE Compliance Detection System

This project is a robust PPE (Personal Protective Equipment) compliance detection system using YOLOv8 for person and PPE detection and Vision Transformer (ViT) for multi-label classification of PPE worn by individuals. It provides a web interface to upload images and visualize detected PPE compliance.


---

Features

Detects persons and their PPE items (helmet, vest, gloves, mask, etc.) in images.

Multi-label classification using ViT for more accurate PPE recognition.

Fusion of YOLO and ViT predictions for a robust compliance score.

Supports automatic annotation of cropped person images.

Web interface for real-time image upload and result visualization.



---

Project Structure

object_detection/
├── app.py                  # Flask web app entry point
├── src/                    # Source code modules
│   ├── train_yolo.py       # YOLOv8 training script
│   ├── train_vit.py        # ViT training script
│   ├── extract_persons.py  # Crop persons from images
│   ├── auto_label_ann.py   # Auto-label PPE on cropped persons
│   ├── fuse_models.py      # Fuse YOLO + ViT predictions
│   └── utils.py            # Utility functions (e.g., folder creation)
├── dataset/                # Dataset folder
│   ├── train/images/       # Training images
│   ├── valid/images/       # Validation images
│   └── test/images/        # Test images
├── weights/                # Saved YOLO and ViT weights
│   ├── yolo_best.pt
│   └── vit_best.pt
├── tmp/                    # Temporary folder
│   ├── persons/            # Cropped person images
│   └── ann.json            # PPE annotations for ViT training
├── static/                 # Web app static assets
│   ├── uploads/            # Uploaded images
│   └── results/            # Result images with predictions
└── README.md               # Project documentation


---

Workflow

1. YOLO Detection

Detects all objects in the image including persons and PPE items.

Crops person regions for ViT processing.



2. ViT Multi-label Classification

Classifies PPE items worn by each person.

Trained on cropped person images and ann.json annotations.



3. Fusion

Combines YOLO bounding boxes and ViT PPE predictions.

Calculates compliance score for each person.

Annotates images with bounding boxes and compliance percentage.



4. Web Interface

Upload images via Flask web app.

View results with colored bounding boxes:

Green: Fully compliant

Orange: Partially compliant

Red: Non-compliant






---

Installation

1. Clone the repository:



git clone https://github.com/rajnishkumar1906/PPE-Detection-Hybrid_YOLOv8_ViT
cd ppe-compliance-detection

2. Create Python environment (Python 3.12 recommended):



conda create -n ppe python=3.12 -y
conda activate ppe

3. Install dependencies:



pip install -r requirements.txt

4. Prepare dataset:



Place images in dataset/train/images/, dataset/valid/images/, dataset/test/images/

Ensure data.yaml is correctly configured with class names and paths.



---

Usage

1. Auto-label PPE annotations for ViT training (optional but recommended):



python src/auto_label_ann.py

Automatically generates tmp/ann.json for ViT training.


2. Run the Flask Web App:



python app.py

Access the web interface at: http://127.0.0.1:5000

Upload images and see PPE compliance results.


3. Training (if needed):



YOLOv8 Training:


python -c "from src.train_yolo import train_yolo; train_yolo('dataset/data.yaml', 'weights', epochs=50)"

ViT Training:


python -c "from src.train_vit import train_vit, PPE_CLASSES; train_vit('tmp/persons', 'tmp/ann.json', 'weights', epochs=15)"


---

Notes

Ensure tmp/persons/ contains cropped person images and ann.json has proper PPE labels for accurate ViT training.

YOLO model (yolo_best.pt) should exist in weights/ to skip retraining.

Adjust ViT epochs for better accuracy; more epochs improve robustness.



---

Supported PPE Classes

PPE_CLASSES = ['helmet', 'vest', 'gloves', 'mask']

Add more classes if needed in both YOLO dataset and ViT annotation file.



---

Contributing

Fork the repository

Create a feature branch

Submit pull requests for improvements, bug fixes, or new PPE classes



---
