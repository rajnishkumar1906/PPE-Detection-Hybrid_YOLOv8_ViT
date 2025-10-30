# src/utils.py

import yaml

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def print_status(msg):
    print(f"[INFO] {msg}")
