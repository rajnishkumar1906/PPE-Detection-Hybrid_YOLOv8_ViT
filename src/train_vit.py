# src/train_vit.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import timm, os, json
from PIL import Image
from src.utils import get_device, ensure_dir
from tqdm import tqdm

PPE_CLASSES = ['helmet', 'vest', 'gloves', 'mask']

class PersonDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, 'r') as f:
            self.ann = json.load(f)
        self.files = list(self.ann.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = os.path.join(self.img_dir, name)
        img = Image.open(img_path).convert("RGB")
        labels = self.ann.get(name, {}).get("ppe_labels", [])
        y = torch.zeros(len(PPE_CLASSES), dtype=torch.float32)
        for i, c in enumerate(PPE_CLASSES):
            if c in labels:
                y[i] = 1.0
        if self.transform:
            img = self.transform(img)
        return img, y

def train_vit(train_dir, ann_file, weights_dir, epochs=5, val_split=0.1, batch_size=8):
    device = get_device()
    ensure_dir(weights_dir)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = PersonDataset(train_dir, ann_file, transform)
    if len(dataset) == 0:
        raise RuntimeError("No ViT training data found. Run extract_persons and create ann.json")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=len(PPE_CLASSES)).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"🚀 Training ViT: train={train_size} val={val_size} epochs={epochs}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        # validation
        model.eval()
        val_acc = 0.0
        batches = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = torch.sigmoid(model(imgs))
                preds_bin = (preds > 0.5).float()
                val_acc += (preds_bin == labels).float().mean().item()
                batches += 1
        val_acc = val_acc / max(batches, 1)
        print(f"Epoch {epoch+1} train_loss={running_loss/len(train_loader):.4f} val_acc={val_acc:.3f}")

    model_path = os.path.join(weights_dir, "vit_best.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ ViT training done. Saved at: {model_path}")
    return model
