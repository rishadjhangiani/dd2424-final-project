import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import get_resnet18
from sklearn.metrics import f1_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["linear_probe", "finetune"], required=True)
parser.add_argument("--label_fraction", type=float, default=1.0)
args = parser.parse_args()

EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader, test_loader = get_dataloaders(
    label_fraction=args.label_fraction
)
if args.mode == "linear_probe":
    model = get_resnet18(num_classes=37, freeze_backbone=True)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    experiment_name = "Linear Probing"

else:
    model = get_resnet18(num_classes=37, freeze_backbone=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    experiment_name = "Fine-Tuning"

model = model.to(device)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

model.eval()
correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = correct / total
macro_f1 = f1_score(all_labels, all_predictions, average="macro")

print(f"Label Fraction: {args.label_fraction}")
print(f"37-Class {experiment_name} Test Accuracy: {test_accuracy:.4f}")
print(f"37-Class {experiment_name} Macro F1: {macro_f1:.4f}")

torch.save(model.state_dict(), "results/models/finetune_model.pth")