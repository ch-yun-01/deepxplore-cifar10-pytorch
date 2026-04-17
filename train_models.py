import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models_torch import CIFARResNet50


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def build_transforms(variant):
    if variant == 1:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif variant == 2:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def set_trainable_layers(model, mode):
    """
    mode 1: full fine-tuning
    mode 2: fc only
    mode 3: layer4 + fc only
    """
    if mode == 1:
        for p in model.parameters():
            p.requires_grad = True

    elif mode == 2:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.fc.parameters():
            p.requires_grad = True

    elif mode == 3:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.layer4.parameters():
            p.requires_grad = True
        for p in model.model.fc.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")


def train_one_model(seed, lr, variant, mode, save_path, epochs=10, batch_size=128):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, test_transform = build_transforms(variant)

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = CIFARResNet50().to(device)
    set_trainable_layers(model, mode)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

    best_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{os.path.basename(save_path)}] trainable params: {trainable_params}/{total_params}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        test_acc = evaluate(model, testloader, device)

        print(
            f"[{os.path.basename(save_path)}] "
            f"epoch={epoch+1}/{epochs} "
            f"loss={running_loss/len(trainloader):.4f} "
            f"test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    print(f"Saved best model to {save_path}, best_acc={best_acc:.4f}")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    # model1: full fine-tune
    train_one_model(
        seed=0,
        lr=0.01,
        variant=1,
        mode=1,
        save_path="checkpoints/model1.pth",
        epochs=25
    )

    # model2: fc only
    train_one_model(
        seed=42,
        lr=0.03,
        variant=2,
        mode=2,
        save_path="checkpoints/model2.pth",
        epochs=25
    )

    # model3: layer4 + fc
    train_one_model(
        seed=123,
        lr=0.01,
        variant=3,
        mode=3,
        save_path="checkpoints/model3.pth",
        epochs=25
    )