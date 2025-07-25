#Mass Map discriminator

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, batch_norm=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.batch_norm = batch_norm

        if self.in_chans != self.out_chans:
            self.out_chans = self.in_chans

        self.conv_1_x_1 = nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(1, 1))
        self.layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = input

        return self.layers(output) + self.conv_1_x_1(output)


class FullDownBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(self.out_chans),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        return self.downsample(input)  

    def __repr__(self):
        return f'AvgPool(in_chans={self.in_chans}, out_chans={self.out_chans}\nResBlock(in_chans={self.out_chans}, out_chans={self.out_chans}'


class DiscriminatorModel(nn.Module):
    def __init__(self, in_chans, out_chans, input_im_size):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.input_im_size = input_im_size

        self.initial_layers = nn.Sequential(
            nn.Conv2d(self.in_chans, 32, kernel_size=(3, 3), padding=1),  # 384x384
            nn.LeakyReLU()
        )

        # This should be refactored to adapt to input and output number of features and the resolution dimensions
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers += [FullDownBlock(32, 64)]
        self.encoder_layers += [FullDownBlock(64, 128)]
        self.encoder_layers += [FullDownBlock(128, 256)]
        self.encoder_layers += [FullDownBlock(256, 512)]
        self.encoder_layers += [FullDownBlock(512, 512)]
        self.encoder_layers += [FullDownBlock(512, 512)]

        downsampled_imsize = self.input_im_size
        for i in range(6):
            downsampled_imsize = downsampled_imsize // 2 # half dimension (rounded down) for every FullDownBlock

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * downsampled_imsize**2 , 1),
        )

    def forward(self, input, y):
        output = torch.cat([input, y], dim=1)
        output = self.initial_layers(output)
        # Apply down-sampling layers
        for layer in self.encoder_layers:
            output = layer(output)
        return self.dense(output)
