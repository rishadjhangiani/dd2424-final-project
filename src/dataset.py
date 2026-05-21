# gets oxford pet dataset, preprocesses and transforms images (cannot directly read raw image files), 
# created test / train / validation splits, and created data loaders to create mini batches

import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42

imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ToTensor(),
    imagenet_normalize,
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    imagenet_normalize,
])

weak_ssl_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ToTensor(),
    imagenet_normalize,
])

strong_ssl_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    imagenet_normalize,
    transforms.RandomErasing(p=0.25),
])


class UnlabeledDataset:
    def __init__(self, base_dataset, weak_transform, strong_transform):
        self.base_dataset = base_dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        weak_image = self.weak_transform(image)
        strong_image = self.strong_transform(image)
        return weak_image, strong_image


def create_subset(dataset, fraction):
    subset_size = int(len(dataset) * fraction)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)


def get_split_indices(dataset_size, train_fraction=0.8):
    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_size = int(train_fraction * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    return train_indices, val_indices


def get_dataloaders(label_fraction=1.0):
    random.seed(RANDOM_SEED)

    train_base_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transforms
    )

    val_base_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=test_transforms
    )

    test_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transforms
    )

    train_indices, val_indices = get_split_indices(len(train_base_dataset))

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    train_dataset = create_subset(train_dataset, label_fraction)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def get_ssl_dataloaders(label_fraction=0.1):
    random.seed(RANDOM_SEED)

    labeled_base_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transforms
    )

    val_base_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=test_transforms
    )

    unlabeled_base_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=None
    )

    test_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transforms
    )

    train_indices, val_indices = get_split_indices(len(labeled_base_dataset))
    labeled_count = int(len(train_indices) * label_fraction)

    labeled_indices = train_indices[:labeled_count]
    unlabeled_indices = train_indices[labeled_count:]

    labeled_dataset = Subset(labeled_base_dataset, labeled_indices)

    unlabeled_subset = Subset(unlabeled_base_dataset, unlabeled_indices)
    unlabeled_dataset = UnlabeledDataset(
        unlabeled_subset,
        weak_ssl_transforms,
        strong_ssl_transforms
    )

    val_dataset = Subset(val_base_dataset, val_indices)

    labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return labeled_loader, unlabeled_loader, val_loader, test_loader