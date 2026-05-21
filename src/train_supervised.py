import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from dataset import get_dataloaders
from models import get_resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["linear_probe", "finetune"], required=True)
parser.add_argument("--label_fraction", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader, test_loader = get_dataloaders(
    label_fraction=args.label_fraction
)

if args.mode == "linear_probe":
    model = get_resnet18(num_classes=37, freeze_backbone=True)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    experiment_name = "Linear Probing"
    method_name = "linear_probe"
else:
    model = get_resnet18(num_classes=37, freeze_backbone=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    experiment_name = "Fine-Tuning"
    method_name = "supervised_finetune"

model = model.to(device)
criterion = nn.CrossEntropyLoss()

best_val_accuracy = 0.0
best_model_path = f"results/models/{method_name}_labels_{args.label_fraction}.pth"

for epoch in range(args.epochs):
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

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch + 1}/{args.epochs}, "
        f"Loss: {avg_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

model.load_state_dict(torch.load(best_model_path, map_location=device))
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
print(f"Saved Best Model: {best_model_path}")

results_path = "results/tables/main_results.csv"
file_exists = os.path.exists(results_path)

with open(results_path, "a", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "method",
            "label_fraction",
            "threshold",
            "epochs",
            "best_val_accuracy",
            "test_accuracy",
            "macro_f1"
        ]
    )

    if not file_exists:
        writer.writeheader()

    writer.writerow({
        "method": method_name,
        "label_fraction": args.label_fraction,
        "threshold": "",
        "epochs": args.epochs,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "macro_f1": macro_f1
    })