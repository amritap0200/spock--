import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.audio_service.model import CRNN
from backend.audio_service.dataset import AudioDataset
import os

torch.set_num_threads(2)
DEVICE = torch.device("cpu")

model = CRNN().to(DEVICE)

# CRNN currently returns sigmoid probabilities; BCELoss expects [0, 1].
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

with torch.no_grad():
    sanity_out = model(torch.zeros(1, 1, 128, 300, device=DEVICE))
    if sanity_out.min().item() < 0.0 or sanity_out.max().item() > 1.0:
        raise ValueError(
            "CRNN output is not in [0, 1]. Switch to BCEWithLogitsLoss "
            "or add sigmoid in CRNN.forward()."
        )



BASE_DIR = Path(__file__).resolve().parent.parent
os.makedirs(BASE_DIR / "weights", exist_ok=True)
DATASET_ROOT = BASE_DIR / "audio_dataset"

REAL_DIR = os.path.join(str(DATASET_ROOT), "real")
FAKE_DIR = os.path.join(str(DATASET_ROOT), "fake")

real_files = [os.path.join(REAL_DIR, f)
              for f in os.listdir(REAL_DIR)
              if f.endswith(".flac")]

fake_files = [os.path.join(FAKE_DIR, f)
              for f in os.listdir(FAKE_DIR)
              if f.endswith(".flac")]

print("Real files:", len(real_files))
print("Fake files:", len(fake_files))

files = real_files + fake_files
labels = [0]*len(real_files) + [1]*len(fake_files)

dataset = AudioDataset(files, labels)
if len(dataset) < 2:
    raise ValueError("Need at least 2 samples to create train/validation split.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
if train_size == 0:
    train_size, val_size = 1, len(dataset) - 1
if val_size == 0:
    val_size, train_size = 1, len(dataset) - 1

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0

    for mel, label in train_loader:
        mel = mel.to(DEVICE)
        label = label.to(DEVICE).unsqueeze(1)

        output = model(mel)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
    torch.save({
       "model_state": model.state_dict(),
       "epoch": epoch
    }, str(BASE_DIR / "weights" / f"audio_checkpoint_epoch_{epoch+1}.pth"))



model.eval()

#test
real_scores = []
fake_scores = []

with torch.no_grad():
    for mel, label in val_loader:
        mel = mel.to(DEVICE)
        output = model(mel).cpu().numpy().flatten()
        
        for o, l in zip(output, label):
            if l.item() == 0:
                real_scores.append(float(o))
            else:
                fake_scores.append(float(o))

import numpy as np

print("Real mean:", np.mean(real_scores))
print("Fake mean:", np.mean(fake_scores))
#test


all_scores = real_scores + fake_scores
threshold = 0
accuracy = 0

for t in np.linspace(0, 1, 200):
    correct = 0

    for s in real_scores:
        if s < t:
            correct += 1

    for s in fake_scores:
        if s >= t:
            correct += 1

    acc = correct / len(all_scores)

    if acc > accuracy:
        accuracy = acc
        threshold = t

torch.save({
    "model_state": model.state_dict(),
    "threshold": threshold
}, str(BASE_DIR / "weights" / "audio_model.pth"))

print("Real mean:", np.mean(real_scores))
print("Fake mean:", np.mean(fake_scores))
print("Best threshold:", threshold)