import argparse
import csv
import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from dataset import get_ssl_dataloaders
from models import get_resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--label_fraction", type=float, default=0.1)
parser.add_argument("--threshold", type=float, default=0.95)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

UNSUPERVISED_WEIGHT = 1.0

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

labeled_loader, unlabeled_loader, val_loader, test_loader = get_ssl_dataloaders(
    label_fraction=args.label_fraction
)

model = get_resnet18(num_classes=37, freeze_backbone=False)
model = model.to(device)

supervised_criterion = nn.CrossEntropyLoss()
unsupervised_criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.0001)

best_val_accuracy = 0.0
best_model_path = (
    f"results/models/ssl_fixmatch_labels_{args.label_fraction}_"
    f"threshold_{args.threshold}.pth"
)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    total_supervised_loss = 0
    total_unsupervised_loss = 0

    unlabeled_cycle = itertools.cycle(unlabeled_loader)

    for labeled_images, labels in labeled_loader:
        weak_images, strong_images = next(unlabeled_cycle)

        labeled_images = labeled_images.to(device)
        labels = labels.to(device)
        weak_images = weak_images.to(device)
        strong_images = strong_images.to(device)

        supervised_outputs = model(labeled_images)
        supervised_loss = supervised_criterion(supervised_outputs, labels)

        with torch.no_grad():
            weak_outputs = model(weak_images)
            probabilities = torch.softmax(weak_outputs, dim=1)
            max_probs, pseudo_labels = torch.max(probabilities, dim=1)
            mask = max_probs.ge(args.threshold).float()

        strong_outputs = model(strong_images)
        unsupervised_losses = unsupervised_criterion(strong_outputs, pseudo_labels)

        if mask.sum() > 0:
            unsupervised_loss = (unsupervised_losses * mask).sum() / mask.sum()
        else:
            unsupervised_loss = torch.tensor(0.0, device=device)

        loss = supervised_loss + UNSUPERVISED_WEIGHT * unsupervised_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_supervised_loss += supervised_loss.item()
        total_unsupervised_loss += unsupervised_loss.item()

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

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch + 1}/{args.epochs}, "
        f"Loss: {total_loss / len(labeled_loader):.4f}, "
        f"Supervised Loss: {total_supervised_loss / len(labeled_loader):.4f}, "
        f"Unsupervised Loss: {total_unsupervised_loss / len(labeled_loader):.4f}, "
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
print(f"Confidence Threshold: {args.threshold}")
print(f"SSL Test Accuracy: {test_accuracy:.4f}")
print(f"SSL Macro F1: {macro_f1:.4f}")
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
        "method": "ssl_fixmatch",
        "label_fraction": args.label_fraction,
        "threshold": args.threshold,
        "epochs": args.epochs,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "macro_f1": macro_f1
    })