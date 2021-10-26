import torch
import torch.nn as nn

class CustomModelMNIST(nn.Module):
    def __init__(self, convolutionalFilters, convolutionalLayers, linearLayers, linearLayerParameters, imageSize):
        super(CustomModelMNIST, self).__init__()

        self.convolutionalFilters = convolutionalFilters
        self.convolutionalLayers = convolutionalLayers
        self.linearLayers = linearLayers
        self.linearLayerParameters = linearLayerParameters
        self.imageSize = imageSize

        ls_conv = []
        ls_lin = []

        filters = 1
        for _ in range(convolutionalLayers):
            ls_conv += [
                nn.Conv2d(filters, self.convolutionalFilters, stride=1, kernel_size=3, padding=1), # doesn't change image size
                nn.InstanceNorm2d(self.convolutionalFilters),
                nn.ReLU(),
            ]
            filters = self.convolutionalFilters
        ls_conv += [ nn.AdaptiveAvgPool2d(1) ]

        for _ in range(linearLayers):
            ls_lin += [
                nn.Linear(filters, self.linearLayerParameters), # doesn't change image size
                nn.LayerNorm(self.linearLayerParameters),
                nn.ReLU(),
            ]
            filters = self.linearLayerParameters
        ls_lin += [ nn.Linear(filters, 10) ] # MNIST classification        

        self.model_conv = nn.Sequential(*ls_conv)
        self.model_lin = nn.Sequential(*ls_lin)

    def forward(self, x):
        N = x.shape[0]
        # ignore actual input, just train on random noise of the correct size
        # (upscaling / downscaling actual MNIST would require CPU usage which we prefer to avoid)
        x = torch.randn((N, 1, self.imageSize, self.imageSize)) # single channel (grayscale)
        x = self.model_conv(x)
        x = x[:, :, 0, 0] # only a single pixel after average pool, removes these two last dimensions
        x = self.model_lin(x)
        return x
