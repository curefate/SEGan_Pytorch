import argparse
import os
from torchvision import utils
import torch
from model import Generator


def save_img(img, path):
    utils.save_image(
        img,
        f"{path}/img_result.png",
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )


def generate(args, device, g_ema):
    styles = None
    latents = None
    if args.styles is not None:
        styles = torch.load(args.styles).to(device)

    if args.latents is not None:
        latents = torch.load(args.latents).to(device)
    else:
        latents = torch.randn(args.num, args.latent_dim, device=device)

    if args.return_styles is True:
        img, ret_style = g_ema(latents, styles, args.return_styles)
        torch.save(ret_style.clone(), f"{args.path}/style_result.pth")
    else:
        img = g_ema(latents, styles, args.return_styles)

    save_img(img, args.path)
    print("done!")
    return


if __name__ == '__main__':
    device = "cuda"

    # <editor-fold desc = "args">
    parser = argparse.ArgumentParser(description="Generate")
    parser.add_argument(
        "ckpt", type=str, help="path to the checkpoint"
    )
    parser.add_argument(
        "--latents", type=str, default=None, help="path to latents(z)(valid only when styles=None)"
    )
    parser.add_argument(
        "--styles", type=str, default=None, help="path to styles(w)"
    )
    parser.add_argument(
        "--num", type=int, default=64, help="number of samples(valid only when latents=None&&styles=None)"
    )
    parser.add_argument(
        "--path", type=str, default="./result", help="result path"
    )
    parser.add_argument(
        "--return_styles", type=bool, default=True, help="whether return the styles"
    )
    parser.add_argument(
        "--resolution", type=int, default=256, help="image resolution for the model"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=512, help="dimensions of latent code"
    )
    args = parser.parse_args()
    # </editor-fold>

    # load model
    g_ema = Generator(args.resolution, args.latent_dim).to(device)
    g_ema.eval()
    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    g_ema.load_state_dict(ckpt["g_ema"])
    for p in g_ema.parameters():
        p.requires_grad = False

    generate(args, device, g_ema)
