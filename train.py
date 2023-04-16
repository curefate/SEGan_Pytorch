import argparse
import os
import random

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainning.dataset import DatasetReader
from trainning.loss import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize
from model import Generator, Discriminator
from model_SEG import Generator_Mode1, Generator_Mode2, Generator_Mode3
from trainning.non_leaking import augment, AdaptiveAugment

from trainning.distributed import (
    get_rank,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


# 调整模型是否接受梯度
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# 清除grad
def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


# 将g累加到g_ema
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


# 用于将loader变成iter
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def make_noise(batch, latent_dim, device):
    noises = torch.randn(batch, latent_dim, device=device)
    return noises


def train(args, loader, generator, discriminator, g_ema, g_optim, d_optim, device):
    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    loader = sample_data(loader)

    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    # ------------------------------------------
    # 图像增强的概率
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    # ------------------------------------------
    # 图像增强
    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    # ------------------------------------------
    # latent code to make sample image for saved
    sample_z = torch.randn(args.n_sample, args.latent_dim, device=device)

    # ------------------------------------------
    # save log
    log_save_path = "./logs/mode" + str(args.mode) + "/"
    if args.log:
        logs = SummaryWriter(log_save_path)

    # ------------------------------------------
    # 正式开始训练
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)[0]
        real_img = real_img.to(device)

        # ------------------------------------------
        # 优化D
        # 关闭G的反馈 打开D的
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = make_noise(args.batch, args.latent_dim, device)
        fake_img = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        # ------------------------------------------
        # 得到loss
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # ------------------------------------------
        # 应用增强
        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        # ------------------------------------------
        # 每d_reg_every次执行一次正则化，lazy regularization
        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # ------------------------------------------
        # 优化G
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = make_noise(args.batch, args.latent_dim, device)
        fake_img = generator(noise)
        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # 每g_reg_every次执行一次正则化，lazy regularization
        g_regularize = i % args.g_reg_every == 0
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = make_noise(args.batch, args.latent_dim, device)
            fake_img, latents = generator(noise, return_styles=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # ------------------------------------------
        # 将g累加到g_ema
        accum = 0.5 ** (32 / (10 * 1000))
        accumulate(g_ema, g_module, accum)

        # ------------------------------------------
        # log title
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        # ------------------------------------------
        # saves
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            # save logs
            if args.log:
                logs.add_scalar(
                    tag='Generator Loss',
                    scalar_value=g_loss_val,
                    global_step=i)
                logs.add_scalar(
                    tag='Discriminator Loss',
                    scalar_value=d_loss_val,
                    global_step=i)
                logs.add_scalar(
                    tag='Augment probability',
                    scalar_value=ada_aug_p,
                    global_step=i)
                logs.add_scalar(
                    tag='Rt',
                    scalar_value=r_t_stat,
                    global_step=i)
                logs.add_scalar(
                    tag='R1',
                    scalar_value=r1_val,
                    global_step=i)
                logs.add_scalar(
                    tag='Path Length Regularization',
                    scalar_value=path_loss_val,
                    global_step=i)
                logs.add_scalar(
                    tag='Mean Path Length',
                    scalar_value=mean_path_length,
                    global_step=i)
                logs.add_scalar(
                    tag='Real Score',
                    scalar_value=real_score_val,
                    global_step=i)
                logs.add_scalar(
                    tag='Fake Score',
                    scalar_value=fake_score_val,
                    global_step=i)
                logs.add_scalar(
                    tag='Path Length',
                    scalar_value=path_length_val,
                    global_step=i)

            # save models
            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample = g_ema(sample_z)
                    logs.add_images('sample', sample, i)
                    sample_save_path = "sample/mode" + str(args.mode) + "/"
                    if not os.path.exists(sample_save_path):
                        os.mkdir(sample_save_path)
                    utils.save_image(
                        sample,
                        sample_save_path + "{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
            if i % 10000 == 0:
                ckpt_save_path = "checkpoints/mode" + str(args.mode) + "/"
                if not os.path.exists(ckpt_save_path):
                    os.mkdir(ckpt_save_path)
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    ckpt_save_path + f"{str(i).zfill(6)}.pt",
                )


if __name__ == '__main__':
    device = "cuda"

    # <editor-fold desc = "args">
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument(
        "path", type=str, help="path to the dataset"
    )
    parser.add_argument(
        "--iter", type=int, default=100000, help="total training iterations"
    )
    parser.add_argument(
        "--start_iter", type=int, default=0, help="start iter"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--resolution", type=int, default=256, help="image resolution for the model"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=512, help="dimensions of latent code"
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--n_sample", type=int, default=8,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize", type=float, default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink", type=int, default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every", type=int, default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every", type=int, default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p", type=float, default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target", type=float, default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length", type=int, default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every", type=int, default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--log", type=bool, default=True, help="Whether to save the log",
    )
    parser.add_argument(
        "--mode", type=int, default=0, help="Generator mode"
    )
    args = parser.parse_args()
    # </editor-fold>

    # ---------------------------------------------
    # set models
    if args.mode == 0:
        generator = Generator(args.resolution, args.latent_dim).to(device)
    elif args.mode == 1:
        generator = Generator_Mode1(args.resolution, args.latent_dim).to(device)
    elif args.mode == 2:
        generator = Generator_Mode2(args.resolution, args.latent_dim).to(device)
    elif args.mode == 3:
        generator = Generator_Mode3(args.resolution, args.latent_dim).to(device)
    discriminator = Discriminator(args.resolution).to(device)
    g_ema = Generator(args.resolution, args.latent_dim).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    # ---------------------------------------------
    # lazy regularize ratio
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    # ---------------------------------------------
    # set optimizer
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # ---------------------------------------------
    # load ckpt
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    # ---------------------------------------------
    # set distribute
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # ---------------------------------------------
    # load dataset
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = DatasetReader(args.path, transform, args.resolution)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    train(args, loader, generator, discriminator, g_ema, g_optim, d_optim, device)
