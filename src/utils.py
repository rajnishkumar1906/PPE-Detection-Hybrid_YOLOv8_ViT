# src/utils.py
import os
import torch

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    return device

