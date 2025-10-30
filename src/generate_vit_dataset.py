# src/generate_vit_dataset.py

import os
import shutil

def prepare_vit_dataset(dataset_path: str, output_path="vit_dataset"):
    """
    Converts YOLO-format dataset to folder-based classification format for ViT.
    Example:
    vit_dataset/
        ├── helmet/
        ├── vest/
        ├── gloves/
        └── boots/
    """
    print("🧩 Preparing dataset for Vision Transformer...")

    classes = ["helmet", "vest", "gloves", "boots"]
    src_images = os.path.join(dataset_path, "train", "images")
    src_labels = os.path.join(dataset_path, "train", "labels")

    if not os.path.exists(src_images) or not os.path.exists(src_labels):
        raise FileNotFoundError("[❌] Missing 'train/images' or 'train/labels' in dataset")

    os.makedirs(output_path, exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(output_path, c), exist_ok=True)

    label_files = [f for f in os.listdir(src_labels) if f.endswith(".txt")]

    for label_file in label_files:
        with open(os.path.join(src_labels, label_file)) as f:
            lines = f.readlines()
        if not lines:
            continue

        class_ids = set([int(line.split()[0]) for line in lines])
        image_name = label_file.replace(".txt", ".jpg")
        src_image = os.path.join(src_images, image_name)

        for cls_id in class_ids:
            if cls_id < len(classes):
                dest_path = os.path.join(output_path, classes[cls_id], image_name)
                if os.path.exists(src_image):
                    shutil.copy(src_image, dest_path)

    print(f"✅ ViT dataset prepared at: {output_path}")
    return output_path