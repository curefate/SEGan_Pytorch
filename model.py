import math
import time

import torch

from torch import nn
from torch.nn import functional as F
from torchvision import utils

from op import conv2d_gradfix
from op.fused_act import FusedLeakyReLU, fused_leaky_relu
from op.upfirdn2d import upfirdn2d


# -------------------------------------------------------------
# <editor-fold desc="Base module">
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()
    return k


# 本质上是FC 会根据in_channel调整weight的scale
class EqualLinear(nn.Module):
    def __init__(self, in_channel, out_channel,
                 bias=True, bias_init=0, lr_mul=1.0, activation=None):
        super(EqualLinear, self).__init__()

        self.activation = activation
        self.scale = (1 / math.sqrt(in_channel)) * lr_mul
        self.lr_mul = lr_mul

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out


# 会用scale调整weight的卷积,用于D
class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


# 上采样时先对feature map采样后模糊
# 下采样时先对feature map模糊后采样
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


# </editor-fold>

# -------------------------------------------------------------
# <editor-fold desc = "SynthesisNetwork">
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class ModulateConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 upsample=False, demodulate=True, blur_kernel=[1, 3, 3, 1]):
        super(ModulateConv, self).__init__()
        self.eps = 1e-8  # where ǫ is a small constant to avoid numerical issues
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.demodulate = demodulate

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulate = EqualLinear(style_dim, in_channel, bias_init=1)

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        # 融合的情况，首先映射w到style，然后缩放和调制weight
        # style  size: batch  1   ic  1   1
        # weight size:   1    oc  ic  ks  ks
        # 原论文公式1
        style = self.modulate(style)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        # 公式3
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        # 卷积
        # 上采样时先对feature map采样后模糊
        # TODO Problem
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim,
                 upsample=False, demodulate=True):
        super(StyleBlock, self).__init__()

        self.conv = ModulateConv(in_channel=in_channel, out_channel=out_channel,
                                 kernel_size=3, style_dim=style_dim,
                                 upsample=upsample, demodulate=demodulate, )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True,
                 blur_kernel=[1, 3, 3, 1]):
        super(ToRGB, self).__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulateConv(in_channel=in_channel, out_channel=3,
                                 kernel_size=1, style_dim=style_dim,
                                 demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class SynthesisBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim):
        super(SynthesisBlock, self).__init__()
        self.in_channel = in_channel
        self.conv_num = 0
        self.torgb_num = 1

        if in_channel != 0:
            self.conv0 = StyleBlock(
                in_channel=in_channel,
                out_channel=out_channel,
                style_dim=style_dim,
                upsample=True
            )
            self.conv_num += 1

        self.conv1 = StyleBlock(
            in_channel=out_channel,
            out_channel=out_channel,
            style_dim=style_dim,
            upsample=False
        )
        self.conv_num += 1

        if in_channel == 0:
            self.torgb = ToRGB(out_channel, style_dim, upsample=False)
        else:
            self.torgb = ToRGB(out_channel, style_dim)

    def forward(self, input, style, noise=None, skip=None):
        style_iter = iter(style.unbind(dim=1))

        if skip is None:
            out = self.conv1(input, next(style_iter))
        else:
            out = self.conv0(input, next(style_iter))
            out = self.conv1(out, next(style_iter))

        skip = self.torgb(out, next(style_iter), skip)
        return out, skip


class SynthesisNetwork(nn.Module):
    def __init__(self, resolution, style_dim,
                 channel_base=32768, channel_max=128):
        super(SynthesisNetwork, self).__init__()
        self.resolution = resolution
        self.style_dim = style_dim

        self.resolution_log2 = int(math.log(resolution, 2))
        self.resolution_list = [2 ** i for i in range(2, self.resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.resolution_list}
        self.style_num = 0 + 1  # last to_rgb layer

        self.const = ConstantInput(self.channels_dict[4])
        self.blocks = nn.ModuleList()

        for res in self.resolution_list:
            in_channel = self.channels_dict[res // 2] if res > 4 else 0
            out_channel = self.channels_dict[res]
            block = SynthesisBlock(in_channel=in_channel,
                                   out_channel=out_channel,
                                   style_dim=style_dim)
            self.blocks.append(block)
            self.style_num += block.conv_num

    def forward(self, styles):
        block_style = []
        style_idx = 0
        for block in self.blocks:
            # block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num).view(-1, self.style_dim))
            block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num))
            style_idx += block.conv_num

        input = self.const(styles)
        out = skip = None
        for block, cur_style in zip(self.blocks, block_style):
            if block.in_channel == 0:
                out, skip = block(input, cur_style)
            else:
                out, skip = block(out, cur_style, skip=skip)

        image = skip
        return image


# </editor-fold>

# -------------------------------------------------------------
# <editor-fold desc="MappingNetwork">
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# TODO TRUNCATION
class MappingNetwork(nn.Module):
    def __init__(self, style_dim, style_num, n_mlp=8, lr_mlp=0.01):
        super(MappingNetwork, self).__init__()
        self.style_num = style_num
        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.layer = nn.Sequential(*layers)

    def forward(self, latents):
        # styles = [self.layer(l) for l in latents]
        # return styles
        styles = self.layer(latents.to(torch.float32))
        styles = styles.unsqueeze(1).repeat([1, self.style_num, 1])
        return styles


# </editor-fold>

# -------------------------------------------------------------
# <editor-fold desc="Generator">
# TODO NOISE
class Generator(nn.Module):
    def __init__(self, resolution=1024, style_dim=512,
                 lr_mlp=0.01):
        super(Generator, self).__init__()
        self.resolution = resolution
        self.style_dim = style_dim

        self.Synthesis = SynthesisNetwork(resolution, style_dim)
        self.Mapping = MappingNetwork(style_dim, self.Synthesis.style_num, lr_mlp=lr_mlp)

    def forward(self, latents, styles=None, return_styles=False):
        if styles is None:
            styles = self.Mapping(latents)

        img = self.Synthesis(styles)
        if return_styles:
            return img, styles
        return img


# </editor-fold>

# -------------------------------------------------------------
# <editor-fold desc="Discriminator">
class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, resolution, channel_base=32768, channel_max=512):
        super().__init__()
        resolution_log2 = int(math.log(resolution, 2))
        self.resolution_list = [2 ** i for i in range(2, resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.resolution_list}

        convs = [ConvLayer(3, self.channels_dict[resolution], 1)]

        in_channel = self.channels_dict[resolution]

        for i in range(resolution_log2, 2, -1):
            out_channel = self.channels_dict[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, self.channels_dict[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(self.channels_dict[4] * 4 * 4, self.channels_dict[4], activation="fused_lrelu"),
            EqualLinear(self.channels_dict[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

# </editor-fold>


if __name__ == '__main__':
    device = 'cuda'
    g = Generator(resolution=256, style_dim=512).to(device)
    print("generator created")
    sample_z = torch.randn(8, 512, device=device)
    sample = g(sample_z)
    utils.save_image(
        sample,
        f"sample.png",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )
    print('done')
