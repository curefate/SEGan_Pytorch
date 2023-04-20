import math
from torchvision import utils
import torch
from torch import nn
from model import MappingNetwork, ConstantInput, StyleBlock, ToRGB
from subspace import EigenSpace, EigenSpace_2style, EigenSpace_onlyL


# <editor-fold desc = "origin(Mode1)">
class SynthesisBlock_Mode1(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, res):
        super(SynthesisBlock_Mode1, self).__init__()
        self.in_channel = in_channel
        self.conv_num = 0
        self.torgb_num = 1

        self.subspaces = nn.ModuleList()

        if in_channel != 0:
            self.conv0 = StyleBlock(
                in_channel=in_channel,
                out_channel=out_channel,
                style_dim=style_dim,
                upsample=True
            )
            self.conv_num += 1
            self.subspaces.append(EigenSpace(res, out_channel, style_dim))

        self.conv1 = StyleBlock(
            in_channel=out_channel,
            out_channel=out_channel,
            style_dim=style_dim,
            upsample=False
        )
        self.conv_num += 1
        self.subspaces.append(EigenSpace(res, out_channel, style_dim))

        if in_channel == 0:
            self.torgb = ToRGB(out_channel, style_dim, upsample=False)
            self.subspaces.append(EigenSpace(res, 3, style_dim))
        else:
            self.torgb = ToRGB(out_channel, style_dim)
            self.subspaces.append(EigenSpace(res, 3, style_dim))

    def forward(self, input, style, sub_factor=None, skip=None):
        style_iter = iter(style.unbind(dim=1))
        if sub_factor is not None:
            for factor, space in zip(sub_factor, self.subspaces):
                space.factor = factor

        sub_idx = 0
        if skip is None:
            temp = next(style_iter)
            out = self.conv1(input, temp)
            out = out + self.subspaces[sub_idx](temp)
            sub_idx += 1
        else:
            temp = next(style_iter)
            out = self.conv0(input, temp)
            out = out + self.subspaces[sub_idx](temp)
            sub_idx += 1
            temp = next(style_iter)
            out = self.conv1(out, temp)
            out = out + self.subspaces[sub_idx](temp)
            sub_idx += 1

        temp = next(style_iter)
        skip = self.torgb(out, temp, skip)
        skip = skip + self.subspaces[sub_idx](temp)
        return out, skip


class SynthesisNetwork_Mode1(nn.Module):
    def __init__(self, resolution, style_dim,
                 channel_base=32768, channel_max=128):
        super(SynthesisNetwork_Mode1, self).__init__()
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
            block = SynthesisBlock_Mode1(in_channel=in_channel,
                                         out_channel=out_channel,
                                         style_dim=style_dim,
                                         res=res)
            self.blocks.append(block)
            self.style_num += block.conv_num

    def forward(self, styles, sub_factor=None):
        block_style = []
        block_factor = []
        style_idx = 0
        for block in self.blocks:
            # block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num).view(-1, self.style_dim))
            block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num))
            block_factor.append(sub_factor[style_idx:block.conv_num + block.torgb_num])
            style_idx += block.conv_num

        input = self.const(styles)
        out = skip = None
        for block, cur_style, cur_factor in zip(self.blocks, block_style, block_factor):
            if block.in_channel == 0:
                out, skip = block(input, cur_style, cur_factor)
            else:
                out, skip = block(out, cur_style, cur_factor, skip=skip)

        image = skip
        return image


class Generator_Mode1(nn.Module):
    def __init__(self, resolution=1024, style_dim=512,
                 lr_mlp=0.01):
        super(Generator_Mode1, self).__init__()
        self.resolution = resolution
        self.style_dim = style_dim

        self.Synthesis = SynthesisNetwork_Mode1(resolution, style_dim)
        self.Mapping = MappingNetwork(style_dim, self.Synthesis.style_num, lr_mlp=lr_mlp)

    def forward(self, latents, sub_factor=None, return_styles=False):
        if sub_factor is None:
            sub_factor = []
            for i in range(self.Synthesis.style_num):
                sub_factor.append(1.)
        styles = self.Mapping(latents)
        img = self.Synthesis(styles, sub_factor)
        if return_styles:
            return img, styles
        return img


# </editor-fold>

# <editor-fold desc = "2style(Mode2)">
# class SynthesisNetwork_Mode2(nn.Module):
#     def __init__(self, resolution, style_dim,
#                  channel_base=32768, channel_max=128):
#         super(SynthesisNetwork_Mode2, self).__init__()
#         self.resolution = resolution
#         self.style_dim = style_dim
#
#         self.resolution_log2 = int(math.log(resolution, 2))
#         self.resolution_list = [2 ** i for i in range(2, self.resolution_log2 + 1)]
#         self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.resolution_list}
#         self.style_num = 0 + 1  # last to_rgb layer
#
#         self.const = ConstantInput(self.channels_dict[4])
#         self.blocks = nn.ModuleList()
#         self.subspaces = nn.ModuleList()
#
#         for res in self.resolution_list:
#             in_channel = self.channels_dict[res // 2] if res > 4 else 0
#             out_channel = self.channels_dict[res]
#             block = SynthesisBlock(in_channel=in_channel,
#                                    out_channel=out_channel,
#                                    style_dim=style_dim)
#             self.blocks.append(block)
#             self.style_num += block.conv_num
#             sub = EigenSpace_2style(res, out_channel, style_dim)
#             self.subspaces.append(sub)
#             sub = EigenSpace_2style(res, out_channel, style_dim)
#             self.subspaces.append(sub)
#
#     def forward(self, styles, sub_factor=None):
#         # subspace
#         if sub_factor is not None:
#             for factor, space in zip(sub_factor, self.subspaces):
#                 space.factor = factor
#         for i in range(self.style_num):
#             styles[i] = self.subspaces[i](styles[i])
#
#         block_style = []
#         style_idx = 0
#         for block in self.blocks:
#             # block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num).view(-1, self.style_dim))
#             block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num))
#             style_idx += block.conv_num
#
#         input = self.const(styles)
#         out = skip = None
#         for block, cur_style in zip(self.blocks, block_style):
#             if block.in_channel == 0:
#                 out, skip = block(input, cur_style)
#             else:
#                 out, skip = block(out, cur_style, skip=skip)
#
#         image = skip
#         return image
#
#
# class Generator_Mode2(nn.Module):
#     def __init__(self, resolution=1024, style_dim=512,
#                  lr_mlp=0.01):
#         super(Generator_Mode2, self).__init__()
#         self.resolution = resolution
#         self.style_dim = style_dim
#
#         self.Synthesis = SynthesisNetwork_Mode2(resolution, style_dim)
#         self.Mapping = MappingNetwork(style_dim, self.Synthesis.style_num, lr_mlp=lr_mlp)
#
#     def forward(self, latents, sub_factor=None, return_styles=False):
#         if sub_factor is None:
#             sub_factor = []
#             for i in range(self.Synthesis.style_num):
#                 sub_factor.append(1.)
#         styles = self.Mapping(latents)
#         img = self.Synthesis(styles, sub_factor)
#         if return_styles:
#             return img, styles
#         return img

class SynthesisBlock_Mode2(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, res):
        super(SynthesisBlock_Mode2, self).__init__()
        self.in_channel = in_channel
        self.conv_num = 0
        self.torgb_num = 1

        self.subspaces = nn.ModuleList()

        if in_channel != 0:
            self.conv0 = StyleBlock(
                in_channel=in_channel,
                out_channel=out_channel,
                style_dim=style_dim,
                upsample=True
            )
            self.conv_num += 1
            self.subspaces.append(EigenSpace_2style(res, out_channel, style_dim))

        self.conv1 = StyleBlock(
            in_channel=out_channel,
            out_channel=out_channel,
            style_dim=style_dim,
            upsample=False
        )
        self.conv_num += 1
        self.subspaces.append(EigenSpace_2style(res, out_channel, style_dim))

        if in_channel == 0:
            self.torgb = ToRGB(out_channel, style_dim, upsample=False)
            self.subspaces.append(EigenSpace_2style(res, 3, style_dim))
        else:
            self.torgb = ToRGB(out_channel, style_dim)
            self.subspaces.append(EigenSpace_2style(res, 3, style_dim))

    def forward(self, input, style, sub_factor=None, skip=None):
        if sub_factor is not None:
            for factor, space in zip(sub_factor, self.subspaces):
                space.factor = factor
        style = style.unbind(dim=1)
        style_list = []
        for i in range(len(style)):
            style_list.append(self.subspaces[i](style[i]))
        style_iter = iter(style_list)

        sub_idx = 0
        if skip is None:
            temp = next(style_iter)
            out = self.conv1(input, temp)
        else:
            temp = next(style_iter)
            out = self.conv0(input, temp)
            temp = next(style_iter)
            out = self.conv1(out, temp)

        temp = next(style_iter)
        skip = self.torgb(out, temp, skip)
        return out, skip


class SynthesisNetwork_Mode2(nn.Module):
    def __init__(self, resolution, style_dim,
                 channel_base=32768, channel_max=128):
        super(SynthesisNetwork_Mode2, self).__init__()
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
            block = SynthesisBlock_Mode2(in_channel=in_channel,
                                         out_channel=out_channel,
                                         style_dim=style_dim,
                                         res=res)
            self.blocks.append(block)
            self.style_num += block.conv_num

    def forward(self, styles, sub_factor=None):
        block_style = []
        block_factor = []
        style_idx = 0
        for block in self.blocks:
            # block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num).view(-1, self.style_dim))
            block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num))
            block_factor.append(sub_factor[style_idx:block.conv_num + block.torgb_num])
            style_idx += block.conv_num

        input = self.const(styles)
        out = skip = None
        for block, cur_style, cur_factor in zip(self.blocks, block_style, block_factor):
            if block.in_channel == 0:
                out, skip = block(input, cur_style, cur_factor)
            else:
                out, skip = block(out, cur_style, cur_factor, skip=skip)

        image = skip
        return image


class Generator_Mode2(nn.Module):
    def __init__(self, resolution=1024, style_dim=512,
                 lr_mlp=0.01):
        super(Generator_Mode2, self).__init__()
        self.resolution = resolution
        self.style_dim = style_dim

        self.Synthesis = SynthesisNetwork_Mode2(resolution, style_dim)
        self.Mapping = MappingNetwork(style_dim, self.Synthesis.style_num, lr_mlp=lr_mlp)

    def forward(self, latents, sub_factor=None, return_styles=False):
        if sub_factor is None:
            sub_factor = []
            for i in range(self.Synthesis.style_num):
                sub_factor.append(1.)
        styles = self.Mapping(latents)
        img = self.Synthesis(styles, sub_factor)
        if return_styles:
            return img, styles
        return img


# </editor-fold>

# <editor-fold desc = "onlyL(Mode3)">
# class Generator_Mode3(nn.Module):
#     def __init__(self, resolution=1024, style_dim=512,
#                  lr_mlp=0.01):
#         super(Generator_Mode3, self).__init__()
#         self.resolution = resolution
#         self.style_dim = style_dim
#
#         self.Synthesis = SynthesisNetwork(resolution, style_dim)
#         self.Mapping = MappingNetwork(style_dim, self.Synthesis.style_num, lr_mlp=lr_mlp)
#         self.Subspaces = nn.ModuleList()
#         for i in range(self.Synthesis.style_num):
#             sub = EigenSpace_onlyL(dim=style_dim)
#             self.Subspaces.append(sub)
#
#     def forward(self, latents, sub_factor=None, return_styles=False):
#         styles = self.Mapping(latents)
#         if sub_factor is None:
#             sub_factor = []
#             for i in range(self.Synthesis.style_num):
#                 sub_factor.append(1.)
#         if sub_factor is not None:
#             for factor, space in zip(sub_factor, self.Subspaces):
#                 space.factor = factor
#         for i in range(self.Synthesis.style_num):
#             styles[i] = self.Subspaces[i](styles[i])
#
#         img = self.Synthesis(styles)
#         if return_styles:
#             return img, styles
#         return img


class SynthesisBlock_Mode3(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, res):
        super(SynthesisBlock_Mode3, self).__init__()
        self.in_channel = in_channel
        self.conv_num = 0
        self.torgb_num = 1

        self.subspaces = nn.ModuleList()

        if in_channel != 0:
            self.conv0 = StyleBlock(
                in_channel=in_channel,
                out_channel=out_channel,
                style_dim=style_dim,
                upsample=True
            )
            self.conv_num += 1
            self.subspaces.append(EigenSpace_onlyL(style_dim))

        self.conv1 = StyleBlock(
            in_channel=out_channel,
            out_channel=out_channel,
            style_dim=style_dim,
            upsample=False
        )
        self.conv_num += 1
        self.subspaces.append(EigenSpace_onlyL(style_dim))

        if in_channel == 0:
            self.torgb = ToRGB(out_channel, style_dim, upsample=False)
            self.subspaces.append(EigenSpace_onlyL(style_dim))
        else:
            self.torgb = ToRGB(out_channel, style_dim)
            self.subspaces.append(EigenSpace_onlyL(style_dim))

    def forward(self, input, style, sub_factor=None, skip=None):
        if sub_factor is not None:
            for factor, space in zip(sub_factor, self.subspaces):
                space.factor = factor
        style = style.unbind(dim=1)
        style_list = []
        for i in range(len(style)):
            style_list.append(self.subspaces[i](style[i]))
        style_iter = iter(style_list)

        if skip is None:
            temp = next(style_iter)
            out = self.conv1(input, temp)
        else:
            temp = next(style_iter)
            out = self.conv0(input, temp)
            temp = next(style_iter)
            out = self.conv1(out, temp)

        temp = next(style_iter)
        skip = self.torgb(out, temp, skip)
        return out, skip


class SynthesisNetwork_Mode3(nn.Module):
    def __init__(self, resolution, style_dim,
                 channel_base=32768, channel_max=128):
        super(SynthesisNetwork_Mode3, self).__init__()
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
            block = SynthesisBlock_Mode3(in_channel=in_channel,
                                         out_channel=out_channel,
                                         style_dim=style_dim,
                                         res=res)
            self.blocks.append(block)
            self.style_num += block.conv_num

    def forward(self, styles, sub_factor=None):
        block_style = []
        block_factor = []
        style_idx = 0
        for block in self.blocks:
            # block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num).view(-1, self.style_dim))
            block_style.append(styles.narrow(1, style_idx, block.conv_num + block.torgb_num))
            block_factor.append(sub_factor[style_idx:block.conv_num + block.torgb_num])
            style_idx += block.conv_num

        input = self.const(styles)
        out = skip = None
        for block, cur_style, cur_factor in zip(self.blocks, block_style, block_factor):
            if block.in_channel == 0:
                out, skip = block(input, cur_style, cur_factor)
            else:
                out, skip = block(out, cur_style, cur_factor, skip=skip)

        image = skip
        return image


class Generator_Mode3(nn.Module):
    def __init__(self, resolution=1024, style_dim=512,
                 lr_mlp=0.01):
        super(Generator_Mode3, self).__init__()
        self.resolution = resolution
        self.style_dim = style_dim

        self.Synthesis = SynthesisNetwork_Mode3(resolution, style_dim)
        self.Mapping = MappingNetwork(style_dim, self.Synthesis.style_num, lr_mlp=lr_mlp)

    def forward(self, latents, sub_factor=None, return_styles=False):
        if sub_factor is None:
            sub_factor = []
            for i in range(self.Synthesis.style_num):
                sub_factor.append(1.)
        styles = self.Mapping(latents)
        img = self.Synthesis(styles, sub_factor)
        if return_styles:
            return img, styles
        return img


# # </editor-fold>

if __name__ == '__main__':
    device = 'cuda'
    g = Generator_Mode3(resolution=16, style_dim=512).to(device)
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