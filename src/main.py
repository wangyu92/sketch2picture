import argparse
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_preparation import CycleDataset
from discriminator import Discriminator
from generator import Generator
from image_pool import ImagePool
from transforms import get_transforms
from utils import save_network, save_outputs


def train_fn(
    disc_X: Discriminator,
    disc_Y: Discriminator,
    gen_X: Generator,
    gen_Y: Generator,
    fake_A_pool: ImagePool,
    fake_B_pool: ImagePool,
    loader: DataLoader,
    opt_disc: optim.Adam,
    opt_gen: optim.Adam,
    lambda_cycle: float,
    lambda_identity: float,
    lambda_paired: float,
    cur_epoch: int,
    sample_dir: str,
    plot_dir: str,
    device: torch.device,
) -> tuple[float | Any, float | Any]:
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    disc_X.train()
    disc_Y.train()
    gen_X.train()
    gen_Y.train()

    loop = tqdm(loader, leave=True)

    running_disc_loss: float = 0.0
    running_gen_loss: float = 0.0

    batch_disc_loss = []
    batch_gen_loss = []

    # X is `Sketch`
    # Y is `Digital`
    # Generator-X takes in a Digital-Image(y) and cvts it to a Sketch-Image
    # Generator-Y takes in a Sketch-Image(y) and cvts it to a Digital-Image
    for idx, (x, y) in enumerate(iterable=loop):
        x: Tensor = x.to(device)
        y: Tensor = y.to(device)

        # Training the Discriminators (Adversarial training so we use MSE Loss)

        # Discriminator_X (works on the X-Samples)
        # Fake sample X generated from the Y->X Generator
        fake_x: Tensor = gen_X(y)
        fake_x = fake_A_pool.query(fake_x)
        # Discriminator predictions on real X samples
        disc_X_real: Tensor = disc_X(x)
        # Discriminator predictions on the fake X samples
        disc_X_fake: Tensor = disc_X(fake_x.detach())

        disc_X_real_loss: Tensor = mse(
            disc_X_real, torch.ones_like(disc_X_real)
        )  # The loss on the real samples by the disc
        disc_X_fake_loss: Tensor = mse(
            disc_X_fake, torch.zeros_like(disc_X_fake)
        )  # The loss on the fake samples by the disc
        disc_X_loss: Tensor = disc_X_real_loss + disc_X_fake_loss

        # Discriminator_Y (works on the Y-Samples)
        # Fake sample Y generated from the X->Y Generator
        fake_y: Tensor = gen_Y(x)
        fake_y = fake_B_pool.query(fake_y)
        # Discriminator predictions on the real Y samples
        disc_Y_real: Tensor = disc_Y(y)
        # Discriminator predictions on the fake Y samples
        disc_Y_fake: Tensor = disc_Y(fake_y.detach())

        # The loss on the real samples by the disc
        disc_Y_real_loss: Tensor = mse(
            disc_Y_real, torch.ones_like(disc_Y_real)
        )
        # The loss on the fake samples by the disc
        disc_Y_fake_loss: Tensor = mse(
            disc_Y_fake, torch.zeros_like(disc_Y_fake)
        )
        disc_Y_loss: Tensor = disc_Y_real_loss + disc_Y_fake_loss

        # Putting it together
        disc_loss: Tensor = (disc_X_loss + disc_Y_loss) / 2

        # Updating the parameters of the discriminators
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Training the Generators (Adversarial training uses MSE and identity-mapping and cycle loss uses L1)
        disc_X_fake = disc_X(
            fake_x
        )  # Predictions by the discriminator on the fake X (Computation included)
        disc_Y_fake = disc_Y(
            fake_y
        )  # Predictions by the discriminator on the fake Y (Computation included)

        # Adversarial loss for the generators
        gen_X_loss = mse(disc_X_fake, torch.ones_like(disc_X_fake))
        gen_Y_loss = mse(disc_Y_fake, torch.ones_like(disc_Y_fake))

        # Cycle consistent loss
        cycle_X = gen_X(fake_y)
        cycle_Y = gen_Y(fake_x)
        cycle_x_loss = l1(x, cycle_X)
        cycle_y_loss = l1(y, cycle_Y)

        # Identity loss
        identity_x = gen_X(x)
        identity_y = gen_Y(y)
        identity_x_loss = l1(x, identity_x)
        identity_y_loss = l1(y, identity_y)

        # Paired Loss
        paired_loss_x = l1(x, fake_x)
        paired_loss_y = l1(y, fake_y)

        # Putting it together
        gen_loss = (
            # Complete Adversarial loss
            gen_X_loss
            + gen_Y_loss
            # Complete Cycle loss
            + cycle_x_loss * lambda_cycle
            + cycle_y_loss * lambda_cycle
            # Complete Identity loss
            + identity_x_loss * lambda_identity
            + identity_y_loss * lambda_identity
            # Complete Paired loss
            + paired_loss_x * lambda_paired
            + paired_loss_y * lambda_paired
        )

        # Updating the parameters of the generators
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if idx == len(loader) - 1:
            # The fake Digital Generated for a real Sketch
            sample_path = pathlib.Path(sample_dir)
            sample_path.mkdir(parents=True, exist_ok=True)
            save_outputs(
                x, fake_y, sample_path / f"RealSketch-FakeDig-{cur_epoch}.png"
            )
            # The fake Sketch Generated for a real Digital
            save_outputs(
                y, fake_x, sample_path / f"RealDig-FakeSketch-{cur_epoch}.png"
            )

        running_disc_loss += disc_loss.item()
        running_gen_loss += gen_loss.item()

        batch_disc_loss.append(disc_loss.item())
        batch_gen_loss.append(gen_loss.item())

        loop.set_description(f"Step [{idx+1}/{len(loader)}]")
        loop.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())

    plot_path = pathlib.Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    plt.plot(batch_disc_loss)
    plt.plot(batch_gen_loss)
    plt.title("Batch Wise Loss Plot")
    plt.legend(["Discriminator Loss", "Generator Loss"])
    plt.tight_layout()
    plt.savefig(plot_path / f"batchwiseLoss-{cur_epoch}.png", dpi=600)
    plt.clf()

    return running_disc_loss / len(loader), running_gen_loss / len(loader)


def inference_fn(
    dataset_dir: str,
    g_s2p: Generator,
    g_p2s: Generator,
    crop_size: int,
    save_dir: str,
    epoch: int,
    device: torch.device,
):
    test_ims_dir = pathlib.Path(dataset_dir) / "test"
    image_paths = list(test_ims_dir.glob("*.jpg")) + list(
        test_ims_dir.glob("*.png")
    )
    transforms = get_transforms(
        load_size=crop_size, crop_size=crop_size, is_train=False
    )

    for image_path in image_paths:
        is_schetched = image_path.stem.startswith("s")

        image = np.array(Image.open(image_path).convert("RGB"))
        image = transforms(image=image)["image"]
        image = image.unsqueeze(0).to(device)
        if is_schetched:
            fake_image = g_s2p(image)
        else:
            fake_image = g_p2s(image)
        save_outputs(
            image,
            fake_image,
            pathlib.Path(save_dir) / f"{image_path.stem}_{epoch}.png",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset_dir", type=str, default="/root/datasets/sketch-digital"
    )
    parser.add_argument("--train_dir", type=str, default="/root/data")
    parser.add_argument("--save_dir", type=str, default="/root/save")
    parser.add_argument("--sample_dir", type=str, default="/root/samples")
    parser.add_argument("--plot_dir", type=str, default="/root/plots")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_size", type=int, default=286)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_identity", type=float, default=0.25)
    parser.add_argument("--lambda_paired", type=float, default=5)
    parser.add_argument("--lambda_cycle", type=float, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_epochs_decay", type=int, default=100)
    parser.add_argument("--gan_mode", type=str, default="lsgan")
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--lr_policy", type=str, default="linear")
    parser.add_argument("--lr_decay_iters", type=int, default=50)
    args: argparse.Namespace = parser.parse_args()

    disc_losses = []
    gen_losses = []

    disc_x: Discriminator = Discriminator(in_channels=3)
    disc_y: Discriminator = Discriminator(in_channels=3)
    gen_y: Generator = Generator(img_channels=3, n_residuals=9)
    gen_x: Generator = Generator(img_channels=3, n_residuals=9)
    disc_x.to(args.device)
    disc_y.to(args.device)
    gen_y.to(args.device)
    gen_x.to(args.device)

    params = list(disc_x.parameters()) + list(disc_y.parameters())
    opt_disc = optim.Adam(
        params=params,
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    params = list(gen_y.parameters()) + list(gen_x.parameters())
    opt_gen = optim.Adam(
        params=params,
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    # Learning rate schedulers
    if args.lr_policy == "linear":

        def lin_fn(x: int) -> float:
            return 1.0 - max(0, x - args.num_epochs_decay) / float(
                args.num_epochs_decay + 1
            )

        scheduler_g = lr_scheduler.LambdaLR(
            optimizer=opt_gen, lr_lambda=lin_fn
        )
        scheduler_d = lr_scheduler.LambdaLR(
            optimizer=opt_disc, lr_lambda=lin_fn
        )
    elif args.lr_policy == "step":
        scheduler_g = lr_scheduler.StepLR(
            optimizer=opt_gen, step_size=args.lr_decay_iters, gamma=0.1
        )
        scheduler_d = lr_scheduler.StepLR(
            optimizer=opt_disc, step_size=args.lr_decay_iters, gamma=0.1
        )
    elif args.lr_policy == "plateau":
        scheduler_g = lr_scheduler.ReduceLROnPlateau(
            optimizer=opt_gen,
            mode="min",
            factor=0.2,
            threshold=0.01,
            patience=5,
        )
        scheduler_d = lr_scheduler.ReduceLROnPlateau(
            optimizer=opt_disc,
            mode="min",
            factor=0.2,
            threshold=0.01,
            patience=5,
        )
    elif args.lr_policy == "cosine":
        scheduler_g = lr_scheduler.CosineAnnealingLR(
            optimizer=opt_gen, T_max=args.num_epochs, eta_min=0
        )
        scheduler_d = lr_scheduler.CosineAnnealingLR(
            optimizer=opt_disc, T_max=args.num_epochs, eta_min=0
        )
    else:
        raise NotImplementedError(
            f"Learning rate policy {args.lr_policy} is not implemented"
        )

    root_x_dir = pathlib.Path(args.dataset_dir) / "Sketches"
    root_y_dir = pathlib.Path(args.dataset_dir) / "Digitals"
    transforms_fn = get_transforms(
        load_size=args.load_size, crop_size=args.crop_size
    )
    dataset = CycleDataset(
        # X-Images are sketches and Y-Images are Digitals
        root_x=root_x_dir,
        root_y=root_y_dir,
        transform=transforms_fn,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    fake_A_pool = ImagePool(args.pool_size)
    fake_B_pool = ImagePool(args.pool_size)

    lrs_g = []
    lsr_d = []
    for epoch in range(args.num_epochs + args.num_epochs_decay):
        lr_g = opt_gen.param_groups[0]["lr"]
        lr_d = opt_disc.param_groups[0]["lr"]
        lrs_g.append(lr_g)
        lsr_d.append(lr_d)

        # Plot the learning rates
        x_data = list(range(len(lrs_g)))
        plot_path = pathlib.Path(args.plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        plt.plot(x_data, lrs_g)
        plt.plot(x_data, lsr_d)
        plt.title("Learning Rate Plot")
        plt.legend(["Generator Learning Rate", "Discriminator Learning Rate"])
        plt.tight_layout()
        plt.savefig(plot_path / "LearningRate.png", dpi=600)
        plt.clf()

        disc_loss, gen_loss = train_fn(
            disc_X=disc_x,
            disc_Y=disc_y,
            gen_X=gen_y,
            gen_Y=gen_x,
            fake_A_pool=fake_A_pool,
            fake_B_pool=fake_B_pool,
            loader=loader,
            opt_disc=opt_disc,
            opt_gen=opt_gen,
            lambda_cycle=args.lambda_cycle,
            lambda_identity=args.lambda_identity,
            lambda_paired=args.lambda_paired,
            cur_epoch=epoch,
            sample_dir=args.sample_dir,
            plot_dir=args.plot_dir,
            device=args.device,
        )

        scheduler_g.step()
        scheduler_d.step()

        disc_losses.append(disc_loss)
        gen_losses.append(gen_loss)

        if args.save_model:
            save_path = pathlib.Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            save_network(
                filename=save_path / f" GeneratorX-{epoch}",
                network=gen_x,
                optimizer=opt_gen,
            )
            save_network(
                filename=save_path / f"GeneratorY-{epoch}",
                network=gen_y,
                optimizer=opt_gen,
            )
            save_network(
                filename=save_path / f"DiscriminatorX-{epoch}",
                network=disc_x,
                optimizer=opt_disc,
            )
            save_network(
                filename=save_path / f"DiscriminatorY-{epoch}",
                network=disc_y,
                optimizer=opt_disc,
            )

        inference_fn(
            dataset_dir=args.dataset_dir,
            g_s2p=gen_x,
            g_p2s=gen_y,
            crop_size=args.crop_size,
            save_dir=args.sample_dir,
            epoch=epoch,
            device=args.device,
        )

    plot_path = pathlib.Path(args.plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    plt.plot(disc_losses)
    plt.plot(gen_losses)
    plt.title(label="Epoch Loss Plot")
    plt.legend(["Discriminator Loss", "Generator Loss"])
    plt.tight_layout()
    plt.savefig(plot_path / "EpochLoss.png", dpi=600)
    plt.clf()


if __name__ == "__main__":
    main()
