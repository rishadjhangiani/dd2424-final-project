from dataset import get_ssl_dataloaders

labeled_loader, unlabeled_loader, val_loader, test_loader = get_ssl_dataloaders(
    label_fraction=0.1
)

print("Labeled batches:", len(labeled_loader))
print("Unlabeled batches:", len(unlabeled_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

labeled_images, labels = next(iter(labeled_loader))
weak_images, strong_images = next(iter(unlabeled_loader))

print("Labeled image shape:", labeled_images.shape)
print("Labels shape:", labels.shape)
print("Weak image shape:", weak_images.shape)
print("Strong image shape:", strong_images.shape)