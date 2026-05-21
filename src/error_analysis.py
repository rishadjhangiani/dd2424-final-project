import argparse
import os

import torch
from torchvision.utils import save_image

from dataset import get_dataloaders
from models import get_resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="results/models/supervised_finetune_labels_1.0.pth")
parser.add_argument("--output_dir", type=str, default="results/error_examples")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.output_dir, exist_ok=True)

_, _, test_loader = get_dataloaders()

model = get_resnet18(num_classes=37, freeze_backbone=False)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

saved_count = 0
max_examples = 20

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        for i in range(len(labels)):
            if predictions[i] != labels[i] and saved_count < max_examples:
                filename = (
                    f"{args.output_dir}/"
                    f"wrong_{saved_count}_pred_{predictions[i].item()}_true_{labels[i].item()}.png"
                )

                save_image(images[i].cpu(), filename)
                saved_count += 1

        if saved_count >= max_examples:
            break

print(f"Saved {saved_count} misclassified examples.")