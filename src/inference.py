import argparse
import pathlib

import numpy as np
import torch
from PIL import Image

from generator import Generator
from transforms import get_transforms
from utils import load_network, save_outputs


def inference_fn(
    dataset_dir: str,
    g_s2p: Generator,
    g_p2s: Generator,
    crop_size: int,
    save_dir: str,
    epoch: int,
    device: torch.device,
) -> None:
    """
    Perform inference on a dataset using two generators, g_s2p and g_p2s, to convert between
    sketches and photos. The function loads images from the test directory of the dataset,
    applies the specified transforms, and generates fake images using the appropriate generator.
    The original and fake images are then saved to the specified directory.

    Args:
        dataset_dir (str): The path to the dataset directory.
        g_s2p (Generator): The generator to use for converting sketches to photos.
        g_p2s (Generator): The generator to use for converting photos to sketches.
        crop_size (int): The size to crop the images to.
        save_dir (str): The path to the directory to save the output images to.
        epoch (int): The epoch number to include in the output image filenames.
        device (torch.device): The device to use for inference.

    Returns:
        None
    """

    g_s2p.eval()
    g_p2s.eval()

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
    parser.add_argument("--save_dir", type=str, default="/root/save")
    parser.add_argument("--sample_dir", type=str, default="/root/samples")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=200)
    args: argparse.Namespace = parser.parse_args()

    gen_s2p_path = (
        pathlib.Path(args.save_dir) / f" GeneratorX-{args.epoch - 1}"
    )
    gen_p2s_path = pathlib.Path(args.save_dir) / f"GeneratorY-{args.epoch - 1}"

    g_s2p = Generator(img_channels=3, n_residuals=9)
    g_p2s = Generator(img_channels=3, n_residuals=9)
    load_network(
        filename=gen_s2p_path,
        network=g_s2p,
        optimizer=None,
        lr=None,
        map_location=args.device,
    )

    load_network(
        filename=gen_p2s_path,
        network=g_p2s,
        optimizer=None,
        lr=None,
        map_location=args.device,
    )
    g_s2p.to(args.device)
    g_p2s.to(args.device)

    inference_fn(
        dataset_dir=args.dataset_dir,
        g_s2p=g_s2p,
        g_p2s=g_p2s,
        crop_size=args.crop_size,
        save_dir=args.sample_dir,
        epoch=200,
        device=args.device,
    )


if __name__ == "__main__":
    main()
