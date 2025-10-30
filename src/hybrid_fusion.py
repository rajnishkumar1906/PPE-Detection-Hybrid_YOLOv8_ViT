# src/hybrid_fusion.py
import torch
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn

def fuse_models(yolo_weights, vit_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔗 Fusing models on {device}")

    yolo = YOLO(yolo_weights).model.to(device)
    vit = models.vit_b_16()
    vit.heads = nn.Linear(vit.heads.head.in_features, 2)
    vit.load_state_dict(torch.load(vit_weights, map_location=device))
    vit = vit.to(device)

    class HybridModel(nn.Module):
        def __init__(self, yolo, vit):
            super().__init__()
            self.yolo = yolo
            self.vit = vit
            self.fc = nn.Linear(2 + 2, 2)

        def forward(self, x):
            yolo_out = self.yolo(x)
            vit_out = self.vit(x)
            fused = torch.cat((yolo_out, vit_out), dim=1)
            return self.fc(fused)

    hybrid = HybridModel(yolo.model, vit)
    print("✅ Hybrid model ready!")
    return hybrid
