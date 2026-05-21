import argparse
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dataset import get_dataloaders
from models import get_resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="results/models/supervised_finetune_labels_1.0.pth")
parser.add_argument("--output_path", type=str, default="results/figures/confusion_matrix.png")
args = parser.parse_args()

os.makedirs("results/figures", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_dataloaders()

model = get_resnet18(num_classes=37, freeze_backbone=False)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(15, 15))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", colorbar=False)

plt.title("37-Class Fine-Tuning Confusion Matrix")
plt.savefig(args.output_path)
plt.close()

print("Saved confusion matrix.")