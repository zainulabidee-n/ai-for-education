import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import time
import psutil
import os

# -----------------------
# Dataset
# -----------------------
class FashionMNISTCSV(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.labels = data.iloc[:, 0].values
        self.images = data.iloc[:, 1:].values

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(28, 28).astype(np.uint8)
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


# -----------------------
# Load data
# -----------------------
dataset = FashionMNISTCSV("fashion-mnist_test.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=False)
torch.save(model.state_dict(), "best_model.pth")
# -----------------------
# Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model = model.to(device)
model.eval()

# -----------------------
# Accuracy
# -----------------------
correct, total = 0, 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = 100 * correct / total

# -----------------------
# Parameters
# -----------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# Inference speed
# -----------------------
start = time.time()

with torch.no_grad():
    for images, _ in loader:
        images = images.to(device)
        _ = model(images)

end = time.time()

inference_time = end - start
images_per_sec = len(dataset) / inference_time

# -----------------------
# Memory usage
# -----------------------
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / (1024 * 1024)

# -----------------------
# Results
# -----------------------
print("Accuracy:", accuracy)
print("Total Parameters:", total_params)
print("Trainable Parameters:", trainable_params)
print("Inference Time (s):", inference_time)
print("Images/sec:", images_per_sec)
print("Memory Usage (MB):", memory_mb)