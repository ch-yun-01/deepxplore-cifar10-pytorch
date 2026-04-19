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
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def build_transforms(variant):
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )

    if variant == 1:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif variant == 2:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, test_tf


def set_trainable_layers(model, mode):
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

        for p in model.model.layer3.parameters():
            p.requires_grad = True
        for p in model.model.layer4.parameters():
            p.requires_grad = True
        for p in model.model.fc.parameters():
            p.requires_grad = True


def train_one_model(seed, lr, variant, mode, pretrained, save_path, epochs=50):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, test_tf = build_transforms(variant)

    trainset = torchvision.datasets.CIFAR10("./data", True, download=True, transform=train_tf)
    testset = torchvision.datasets.CIFAR10("./data", False, download=True, transform=test_tf)

    trainloader = torch.utils.data.DataLoader(trainset, 128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, 128)

    model = CIFARResNet50(pretrained=pretrained).to(device)
    set_trainable_layers(model, mode)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20],
        gamma=0.1
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience = 5
    counter = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # scheduler.step()

        acc = evaluate(model, testloader, device)

        print(
            f"{save_path} epoch {epoch+1} "
            f"acc {acc:.4f} "
            f"lr {optimizer.param_groups[0]['lr']:.5f}"
        )

        if acc > best_acc:
            best_acc = acc
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    # # model1: scratch
    train_one_model(42, 0.005, 1, 1, False, "checkpoints/model1.pth")

    # # model2: pretrained full fine-tune
    train_one_model(42, 0.005, 2, 1, True, "checkpoints/model2.pth")

    # pretrained partial fine-tune
    train_one_model(42, 0.005, 3, 3, True, "checkpoints/model4.pth")
