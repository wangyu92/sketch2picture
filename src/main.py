import argparse
import pathlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from data_preparation import CycleDataset
from discriminator import Discriminator
from generator import Generator
from utils import save_network, save_outputs


def train_fn(
    args, disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, l1, mse
):
    loop = tqdm(loader, leave=True)

    running_disc_loss = 0.0
    running_gen_loss = 0.0

    batch_disc_loss = []
    batch_gen_loss = []

    # X is `Sketch`
    # Y is `Digital`
    # Generator-X takes in a Digital-Image(y) and cvts it to a Sketch-Image
    # Generator-Y takes in a Sketch-Image(y) and cvts it to a Digital-Image
    for idx, (x, y) in enumerate(iterable=loop):
        x = x.to(args.device)
        y = y.to(args.device)

        # Training the Discriminators (Adversarial training so we use MSE Loss)

        # Discriminator_X (works on the X-Samples)
        fake_x = gen_X(y)  # Fake sample X generated from the Y->X Generator
        disc_X_real = disc_X(x)  # Discriminator predictions on real X samples
        disc_X_fake = disc_X(
            fake_x.detach()
        )  # Discriminator predictions on the fake X samples

        disc_X_real_loss = mse(
            disc_X_real, torch.ones_like(disc_X_real)
        )  # The loss on the real samples by the disc
        disc_X_fake_loss = mse(
            disc_X_fake, torch.zeros_like(disc_X_fake)
        )  # The loss on the fake samples by the disc
        disc_X_loss = disc_X_real_loss + disc_X_fake_loss

        # Discriminator_Y (works on the Y-Samples)
        fake_y = gen_Y(x)  # Fake sample Y generated from the X->Y Generator
        disc_Y_real = disc_Y(
            y
        )  # Discriminator predictions on the real Y samples
        disc_Y_fake = disc_Y(
            fake_y.detach()
        )  # Discriminator predictions on the fake Y samples

        disc_Y_real_loss = mse(
            disc_Y_real, torch.ones_like(disc_Y_real)
        )  # The loss on the real samples by the disc
        disc_Y_fake_loss = mse(
            disc_Y_fake, torch.zeros_like(disc_Y_fake)
        )  # The loss on the fake samples by the disc
        disc_Y_loss = disc_Y_real_loss + disc_Y_fake_loss

        # Putting it together
        disc_loss = (disc_X_loss + disc_Y_loss) / 2

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
            + cycle_x_loss * args.lambda_cycle
            + cycle_y_loss * args.lambda_cycle
            # Complete Identity loss
            + identity_x_loss * args.lambda_identity
            + identity_y_loss * args.lambda_identity
            # Complete Paired loss
            + paired_loss_x * args.lambda_paired
            + paired_loss_y * args.lambda_paired
        )

        # Updating the parameters of the generators
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if idx % 200 == 0:
            # The fake Digital Generated for a real Sketch
            sample_path = pathlib.Path(args.sample_dir)
            sample_path.mkdir(parents=True, exist_ok=True)
            save_outputs(
                x, fake_y, sample_path / f"RealSketch-FakeDig-{idx}.png"
            )
            # The fake Sketch Generated for a real Digital
            save_outputs(
                y, fake_x, sample_path / f"RealDig-FakeSketch-{idx}.png"
            )

        running_disc_loss += disc_loss.item()
        running_gen_loss += gen_loss.item()

        batch_disc_loss.append(disc_loss.item())
        batch_gen_loss.append(gen_loss.item())

        loop.set_description(f"Step [{idx+1}/{len(loader)}]")
        loop.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())

        plot_path = pathlib.Path(args.plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)

        plt.plot(batch_disc_loss)
        plt.plot(batch_gen_loss)
        plt.title("Batch Wise Loss Plot")
        plt.legend(["Discriminator Loss", "Generator Loss"])
        plt.tight_layout()
        plt.savefig(plot_path / f"batchwiseLoss-{idx}.png", dpi=600)

    return running_disc_loss / len(loader), running_gen_loss / len(loader)


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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lambda_identity", type=float, default=0.25)
    parser.add_argument("--lambda_paired", type=float, default=5)
    parser.add_argument("--lambda_cycle", type=float, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_model", type=bool, default=True)
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

    opt_disc = optim.Adam(
        params=list(disc_x.parameters()) + list(disc_y.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        params=list(gen_y.parameters()) + list(gen_x.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    root_x_dir = pathlib.Path(args.dataset_dir) / "Sketches"
    root_y_dir = pathlib.Path(args.dataset_dir) / "Digitals"
    dataset = CycleDataset(
        # X-Images are sketches and Y-Images are Digitals
        root_x=root_x_dir,
        root_y=root_y_dir,
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    for epoch in range(args.num_epochs):
        disc_loss, gen_loss = train_fn(
            args,
            disc_x,
            disc_y,
            gen_y,
            gen_x,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
        )

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

    plot_path = pathlib.Path(args.plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    plt.plot(disc_losses)
    plt.plot(gen_losses)
    plt.title(label="Epoch Loss Plot")
    plt.legend(["Discriminator Loss", "Generator Loss"])
    plt.tight_layout()
    plt.savefig(plot_path / "EpochLoss.png", dpi=600)


if __name__ == "__main__":
    main()
