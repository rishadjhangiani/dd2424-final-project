# gets oxford pet dataset, preprocesses and transforms images (cannot directly read raw image files), 
# created test / train / validation splits, and created data loaders to create mini batches


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Image size
IMAGE_SIZE = 224

# Batch size
BATCH_SIZE = 32

# Training transforms
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ToTensor(),
])

# Validation/test transforms
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def get_dataloaders():
    """
    Creates train, validation, and test dataloaders
    using Oxford-IIIT Pet Dataset.
    """

    # Full training dataset
    full_train_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transforms
    )

    # Test dataset
    test_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transforms
    )

    # Create validation split
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # Validation should use test transforms
    val_dataset.dataset.transform = test_transforms

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader