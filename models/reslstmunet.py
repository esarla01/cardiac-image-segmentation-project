#-*- encoding=utf-8 -*-
import torch
import torch.nn as nn
from models.resnet_layer import resnet18
from models.clstm import CLSTM


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiscaleAttConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleAttConvBlock, self).__init__()

        self.conv30 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv31 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv32 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv33 = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.selayer = SELayer(channel=out_channels)

    def forward(self, x):
        x0 = self.conv30(x)
        x1 = self.conv31(x0)
        x2 = self.conv32(x1)
        x3 = self.conv33(torch.cat([x0, x1, x2], dim=1))
        x4 = self.selayer(x3)
        x_shortcut = self.conv11(x)
        x = x4 + x_shortcut
        # x = x3 + x_shortcut
        return  x


class TransConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, multiscale_att=False):
        super(Decoder, self).__init__()

        self.transconv4 = TransConv(512, 256)
        self.transconv3 = TransConv(256, 128)
        self.transconv2 = TransConv(128, 64)
        self.transconv1 = TransConv(64, 64)
        self.transconv0 = TransConv(64, 32)

        if multiscale_att:
            self.decoder_layer4 = MultiscaleAttConvBlock(512, 256)
            self.decoder_layer3 = MultiscaleAttConvBlock(256, 128)
            self.decoder_layer2 = MultiscaleAttConvBlock(128, 64)
            self.decoder_layer1 = MultiscaleAttConvBlock(128, 64)
            self.decoder_layer0 = MultiscaleAttConvBlock(32, 16)

        else:
            self.decoder_layer4 = ConvBlock(512, 256)
            self.decoder_layer3 = ConvBlock(256, 128)
            self.decoder_layer2 = ConvBlock(128, 64)
            self.decoder_layer1 = ConvBlock(128, 64)
            self.decoder_layer0 = ConvBlock(32, 16)


        self.output_layer0 = nn.Conv2d(16, out_channels, kernel_size=1)
        self.output_layer1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.output_layer2 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.output_layer3 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.output_layer4 = nn.Conv2d(256, out_channels, kernel_size=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x0, x1, x2, x3, x4):

        out = self.transconv4(x4)  # 1, 256, 14, 14
        out = torch.cat([x3, out], dim=1)  # 1, 512, 14, 14

        out = self.decoder_layer4(out)  # 1, 256, 14, 14
        out4 = self.output_layer4(out)  # 1, 1, 14, 14

        out = self.transconv3(out)  # 1, 128, 28, 28
        out = torch.cat([x2, out], dim=1)  # 1, 256, 28, 28

        out = self.decoder_layer3(out)  # 1, 128, 28, 28
        out3 = self.output_layer3(out)  # 1, 1, 28, 28

        out = self.transconv2(out)  # 1, 64, 56, 56
        out = torch.cat([x1, out], dim=1)  # 1, 128, 56, 56

        out = self.decoder_layer2(out)  # 1, 64, 56, 56
        out2 = self.output_layer2(out)  # 1, 1, 56, 56

        out = self.transconv1(out)  # 1, 64, 112, 112
        out = torch.cat([x0, out], dim=1)  # 1, 128, 112, 112
        out = self.decoder_layer1(out)   # 1, 64, 112, 112
        out1 = self.output_layer1(out)  # 1, 1, 112, 112

        out = self.transconv0(out)  # 1, 32, 224, 224
        out = self.decoder_layer0(out)  # 1, 16, 224, 224
        out = self.output_layer0(out)  # 1, 1, 224, 224

        return out, out1, out2, out3, out4

class ResLSTMUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=False, deep_sup=False, multiscale_att=False):
        super(ResLSTMUNet, self).__init__()

        self.encoder = resnet18(pretrained=pretrained)
        self.decoder = Decoder(out_channels=out_channels, multiscale_att=multiscale_att)
        self.deep_sup = deep_sup
        self.clstsm0 = CLSTM(64, 64)
        self.clstsm1 = CLSTM(64, 64)
        self.clstsm2 = CLSTM(128, 128)
        self.clstsm3 = CLSTM(256, 256)
        self.clstsm4 = CLSTM(512, 512)

        if not pretrained:
            self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x_serial):
        x0_serial = list()
        x1_serial = list()
        x2_serial = list()
        x3_serial = list()
        x4_serial = list()

        for x in x_serial:
            x0, x1, x2, x3, x4 = self.encoder(x)
            x0_serial.append(x0)
            x1_serial.append(x1)
            x2_serial.append(x2)
            x3_serial.append(x3)
            x4_serial.append(x4)

        x0_lstm = torch.stack(x0_serial, dim=1)
        x1_lstm = torch.stack(x1_serial, dim=1)
        x2_lstm = torch.stack(x2_serial, dim=1)
        x3_lstm = torch.stack(x3_serial, dim=1)
        x4_lstm = torch.stack(x4_serial, dim=1)

        x0_lstm = self.clstsm0(x0_lstm)
        x1_lstm = self.clstsm1(x1_lstm)
        x2_lstm = self.clstsm2(x2_lstm)
        x3_lstm = self.clstsm3(x3_lstm)
        x4_lstm = self.clstsm4(x4_lstm)

        out_serial = list()
        out1_serial = list()
        out2_serial = list()
        out3_serial = list()
        out4_serial = list()

        temporal = x0_lstm.shape[1]
        for t in range(temporal):
            x0 = x0_lstm[:, t, :, :]
            x1 = x1_lstm[:, t, :, :]
            x2 = x2_lstm[:, t, :, :]
            x3 = x3_lstm[:, t, :, :]
            x4 = x4_lstm[:, t, :, :]

            out, out1, out2, out3, out4 = self.decoder(x0, x1, x2, x3, x4)

            out_serial.append(out)
            out1_serial.append(out1)
            out2_serial.append(out2)
            out3_serial.append(out3)
            out4_serial.append(out4)

        if self.deep_sup:
            return out_serial, out1_serial, out2_serial, out3_serial, out4_serial
        else:
            return out_serial