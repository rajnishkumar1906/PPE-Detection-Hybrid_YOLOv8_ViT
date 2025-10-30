# src/train_vit.py
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os

def train_vit_model(dataset_path, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training ViT on: {device}")

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.vit_b_16(weights="IMAGENET1K_V1")
    model.heads = nn.Linear(model.heads.head.in_features, len(train_data.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"✅ Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    os.makedirs("runs/vit", exist_ok=True)
    torch.save(model.state_dict(), "runs/vit/best_vit.pth")
    print("💾 Saved ViT model at runs/vit/best_vit.pth")

    return "runs/vit/best_vit.pth"
