import os
import torch
from torchvision.utils import save_image

from dataset import get_dataloaders
from models import get_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results/error_examples", exist_ok=True)

_, _, test_loader = get_dataloaders()

model = get_resnet18(num_classes=37, freeze_backbone=False)
model.load_state_dict(torch.load("results/models/finetune_model.pth", map_location=device))
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
                    f"results/error_examples/"
                    f"wrong_{saved_count}_pred_{predictions[i].item()}_true_{labels[i].item()}.png"
                )

                save_image(images[i].cpu(), filename)
                saved_count += 1

        if saved_count >= max_examples:
            break

print(f"Saved {saved_count} misclassified examples.")