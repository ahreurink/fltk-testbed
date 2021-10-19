import torch.nn as nn
from fltk.util.config.arguments import LearningParameters

class FashionMNISTCNN(nn.Module):

    def __init__(self,learning_params):
        super(FashionMNISTCNN, self).__init__()
        self.conv_filters = learning_params.conv_filters
        self.conv_layers = learning_params.conv_layers
        self.lin_layers = learning_params.lin_layers
        self.lin_pars = learning_params.lin_pars

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.conv_layers, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.conv_layers),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layern = nn.Sequential(
            nn.Conv2d(self.conv_layers, self.conv_layers, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.conv_layers),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        for _ in self.conv_filters:
            x = self.layern(x)
        x = self.fc(self.flatten(x))

        return x
