import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders
from models import get_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader = get_dataloaders()

# Load model
model = get_resnet18(num_classes=37, freeze_backbone=False)
model.load_state_dict(torch.load("results/models/finetune_model.pth"))
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

plt.savefig("results/figures/confusion_matrix.png")
plt.close()

print("Saved confusion matrix.")