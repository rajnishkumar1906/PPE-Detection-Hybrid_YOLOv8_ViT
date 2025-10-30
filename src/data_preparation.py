# src/data_preparation.py

from pathlib import Path
from src.utils import read_yaml, write_yaml, print_status

def verify_dataset_structure(dataset_path="dataset"):
    """
    Verifies that the dataset directory contains all required subfolders.
    Expected:
      train/images, train/labels
      valid/images, valid/labels
      test/images,  test/labels
    """
    dataset_path = Path(dataset_path)
    expected = [
        dataset_path / "train/images",
        dataset_path / "train/labels",
        dataset_path / "valid/images",
        dataset_path / "valid/labels",
        dataset_path / "test/images",
        dataset_path / "test/labels",
    ]

    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print("\n[❌] Missing dataset folders:")
        for m in missing:
            print("   ", m)
        return False

    print_status("✅ Dataset structure verified successfully.")
    return True


def check_yaml_config(yaml_path="dataset/data.yaml"):
    """
    Verifies and updates the data.yaml file so that all paths are absolute.
    Also ensures 'val' key exists instead of 'valid'.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"[❌] data.yaml not found at: {yaml_path}")

    # Load YAML content
    data = read_yaml(yaml_path)
    base_path = yaml_path.parent.resolve()

    # Convert "valid" key -> "val" if necessary
    if "valid" in data and "val" not in data:
        data["val"] = data.pop("valid")

    # Ensure all paths are absolute
    for key in ["train", "val", "test"]:
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str((base_path / data[key]).resolve())

    # Save back updated YAML
    write_yaml(yaml_path, data)
    print_status(f"💾 Updated YAML paths in {yaml_path}")
    return data


if __name__ == "__main__":
    dataset_path = "dataset"
    yaml_path = f"{dataset_path}/data.yaml"

    print("🔍 Verifying dataset structure...")
    verify_dataset_structure(dataset_path)

    print("\n📄 Checking YAML configuration...")
    check_yaml_config(yaml_path)