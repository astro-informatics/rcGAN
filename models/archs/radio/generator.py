#Mass Map Generator

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU()
        )
        self.conv_1x1 = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        return self.conv_1x1(x) + self.conv_block(x)


class ConvDownBlock(nn.Module):
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

        self.conv_1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.res = ResidualBlock(out_chans)
        self.conv_3 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(out_chans)
        self.activation = nn.PReLU()

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        if self.batch_norm:
            out = self.activation(self.bn(self.conv_1(input)))
            skip_out = self.res(out) 
            out = self.conv_3(skip_out)
        else:
            out = self.activation(self.conv_1(input))
            skip_out = self.res(out)  
            out = self.conv_3(skip_out)

        return out, skip_out


class ConvUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_1 = nn.ConvTranspose2d(in_chans // 2, in_chans // 2, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(in_chans // 2)
        self.activation = nn.PReLU()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.PReLU(),
            ResidualBlock(out_chans),
        )

    def forward(self, input, skip_input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        residual_skip = skip_input  
        upsampled = self.activation(self.bn(self.conv_1(input, output_size=residual_skip.size())))
        concat_tensor = torch.cat([residual_skip, upsampled], dim=1)

        return self.layers(concat_tensor)

    
class ConvUpBlock_alt_upsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.conv_1 = nn.Conv2d(in_chans // 2, in_chans//2, kernel_size=3, padding=1)
            
        self.bn = nn.BatchNorm2d(in_chans // 2)
        self.activation = nn.PReLU()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.PReLU(),
            ResidualBlock(out_chans),
        )

    def forward(self, input, skip_input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        
        residual_skip = skip_input  
        resized = torch.nn.functional.interpolate(input, size=skip_input.size()[2:], mode='nearest')
        upsampled = self.activation(self.bn(self.conv_1(resized)))
        concat_tensor = torch.cat([residual_skip, upsampled], dim=1)

        return self.layers(concat_tensor)


class UNetModel(nn.Module):
    def __init__(self, in_chans, out_chans, alt_upsample=False, chans=128, num_pool_layers=4):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvDownBlock(in_chans, self.chans, batch_norm=False)])
        for i in range(self.num_pool_layers - 1):
            if i < 3:
                self.down_sample_layers += [ConvDownBlock(self.chans, self.chans * 2)]
                self.chans *= 2
            else:
                self.down_sample_layers += [ConvDownBlock(self.chans, self.chans)]

        self.res_layer_1 = nn.Sequential(
            nn.Conv2d(self.chans, self.chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.chans),
            nn.PReLU(),
            ResidualBlock(self.chans),
            ResidualBlock(self.chans),
            ResidualBlock(self.chans),
            ResidualBlock(self.chans),
            ResidualBlock(self.chans),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.chans, self.chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.chans),
            nn.PReLU(),
        )

        self.up_sample_layers = nn.ModuleList()
        
        
        for i in range(self.num_pool_layers - 1):
            if not alt_upsample:
                self.up_sample_layers += [ConvUpBlock(self.chans * 2, self.chans // 2)]
            else:
                self.up_sample_layers += [ConvUpBlock_alt_upsample(self.chans * 2, self.chans // 2)]

            self.chans //= 2

        if not alt_upsample:
            self.up_sample_layers += [ConvUpBlock(self.chans * 2, self.chans)]
        else:
            self.up_sample_layers += [ConvUpBlock_alt_upsample(self.chans * 2, self.chans)]
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.chans, self.chans // 2, kernel_size=1),
            nn.Conv2d(self.chans // 2, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = input
        stack = []
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output, skip_out = layer(output)
            stack.append(skip_out)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = layer(output, stack.pop())

        return self.conv2(output)
