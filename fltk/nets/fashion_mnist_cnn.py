import torch.nn as nn


class FashionMNISTCNN(nn.Module):

    def __init__(self, layerSize, layerAmount):
        super(FashionMNISTCNN, self).__init__()
        self.layerSize = layerSize
        self.layerAmount = layerAmount

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.layerSize, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.layerSize),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layern = nn.Sequential(
            nn.Conv2d(self.layerSize, self.layerSize, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.layerSize),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        for _ in self.Amount:
            x = self.layern(x)
        x = self.fc(self.flatten(x))

        return x
