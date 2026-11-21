
import torch
import torch.nn as nn
from torchvision import models


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = UNetDoubleConv(3, 32)
        self.enc2 = UNetDoubleConv(32, 64)
        self.enc3 = UNetDoubleConv(64, 128)
        self.enc4 = UNetDoubleConv(128, 256)

        self.bottleneck = UNetDoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = UNetDoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = UNetDoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetDoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetDoubleConv(64, 32)

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


def load_classification_model(checkpoint_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def load_segmentation_model(checkpoint_path, device):
    model = UNet()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model
