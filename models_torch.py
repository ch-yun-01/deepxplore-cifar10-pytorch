import torch
import torch.nn as nn
import torchvision.models as models


class CIFARResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # pretrained 사용
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # CIFAR 맞게 수정
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        # classifier 수정
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def forward_with_features(self, x):
        features = {}

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features["relu"] = x

        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        features["layer1"] = x

        x = self.model.layer2(x)
        features["layer2"] = x

        x = self.model.layer3(x)
        features["layer3"] = x

        x = self.model.layer4(x)
        features["layer4"] = x

        x = self.model.avgpool(x)
        features["avgpool"] = x

        x = torch.flatten(x, 1)
        logits = self.model.fc(x)

        return logits, features


def load_model(ckpt_path, device, num_classes=10):
    model = CIFARResNet50(num_classes=num_classes)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model