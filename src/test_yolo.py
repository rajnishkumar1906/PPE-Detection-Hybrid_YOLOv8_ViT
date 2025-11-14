# src/test_yolo.py
def test_yolo(model, image_path, conf=0.4, save=False):
    """
    Run YOLO inference and optionally save.
    """
    print(f"🧪 Running YOLO on: {image_path} (conf={conf})")
    results = model.predict(source=image_path, conf=conf, save=save)
    print("✅ YOLO inference complete.")
    return results
