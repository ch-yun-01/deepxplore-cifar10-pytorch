# eval_accuracy.py
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

    trainset = torchvision.datasets.CIFAR10("./data", True,  download=True, transform=train_tf)
    testset  = torchvision.datasets.CIFAR10("./data", False, download=True, transform=test_tf)

    trainloader = torch.utils.data.DataLoader(trainset, 128, shuffle=True)
    testloader  = torch.utils.data.DataLoader(testset,  128)

    model = CIFARResNet50(pretrained=pretrained).to(device)
    set_trainable_layers(model, mode)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience = 5
    counter  = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        acc = evaluate(model, testloader, device)
        print(f"{save_path} epoch {epoch+1} acc {acc:.4f} lr {optimizer.param_groups[0]['lr']:.5f}")

        if acc > best_acc:
            best_acc = acc
            counter  = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


def load_and_eval(save_path, pretrained, device):
    """저장된 pth를 불러와 test accuracy를 반환한다."""
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset    = torchvision.datasets.CIFAR10("./data", False, download=True, transform=test_tf)
    testloader = torch.utils.data.DataLoader(testset, 256)

    model = CIFARResNet50(pretrained=pretrained).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    return evaluate(model, testloader, device)


def get_model_acc(name, seed, lr, variant, mode, pretrained, save_path, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(save_path):
        print(f"[{name}] checkpoint found — loading {save_path}")
    else:
        print(f"[{name}] checkpoint not found — training...")
        train_one_model(seed, lr, variant, mode, pretrained, save_path, epochs)

    acc = load_and_eval(save_path, pretrained, device)
    print(f"[{name}] test accuracy: {acc * 100:.2f}%")
    return acc


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    configs = [
        ("model1", 42, 0.005, 1, 1, False, "checkpoints/model1.pth"),
        ("model2", 42, 0.005, 2, 1, True,  "checkpoints/model2.pth"),
        ("model3", 42, 0.005, 3, 3, True,  "checkpoints/model3.pth"),
    ]

    results = {}
    for name, seed, lr, variant, mode, pretrained, save_path in configs:
        results[name] = get_model_acc(name, seed, lr, variant, mode, pretrained, save_path)

    print("\n=== Final Results ===")
    for name, acc in results.items():
        print(f"  {name}: {acc * 100:.2f}%")