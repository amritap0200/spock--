import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-3

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR / "video_dataset")
WEIGHTS_DIR = BASE_DIR / "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
if len(dataset) < 2:
    raise ValueError("Need at least 2 samples to create train/validation split.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
if train_size == 0:
    train_size, val_size = 1, len(dataset) - 1
if val_size == 0:
    val_size, train_size = 1, len(dataset) - 1

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, 1)

model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)



print("Starting training...")

for epoch in range(EPOCHS):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")
    torch.save({
       "model_state": model.state_dict(),
       "epoch": epoch
   }, str(WEIGHTS_DIR / f"video_checkpoint_epoch_{epoch+1}.pth"))

print("Training complete.")

model.eval()

real_scores = []
fake_scores = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()

        for p, l in zip(probs, labels):
            score = float(p[0])
            if l.item() == 0:
                real_scores.append(score)
            else:
                fake_scores.append(score)

import numpy as np

all_scores = real_scores + fake_scores
best_threshold = 0
best_accuracy = 0

for t in np.linspace(0, 1, 200):
    correct = 0

    for s in real_scores:
        if s < t:
            correct += 1

    for s in fake_scores:
        if s >= t:
            correct += 1

    acc = correct / len(all_scores)

    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = t

print("Best threshold:", best_threshold)
print("Validation accuracy:", best_accuracy)

# Save weights
os.makedirs(str(BASE_DIR / "weights"), exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "threshold": best_threshold
}, str(WEIGHTS_DIR / "video_model.pth"))

print("Real mean:", np.mean(real_scores))
print("Fake mean:", np.mean(fake_scores))
print("Best threshold:", best_threshold)
print("Model saved.")
