import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid, save_image


def save_network(
    filename: str, network: nn.Module, optimizer: optim.Optimizer, **kwargs
):
    checkpoint = {
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    for param in kwargs:
        checkpoint[param] = kwargs[param]
    print(f"-> Saving Model at {filename}")
    torch.save(checkpoint, filename)


def load_network(
    filename: str,
    network: nn.Module,
    optimizer: optim.Optimizer = None,
    lr: float = None,
    map_location: str = "cpu",
    **kwargs,
):
    checkpoint = torch.load(filename, map_location)
    network.load_state_dict(checkpoint["network"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    meta_data = {}
    for param in kwargs:
        if checkpoint.get(param, None) is not None:
            meta_data[param] = checkpoint[param]

    return meta_data


def save_outputs(sketch: torch.Tensor, digital: torch.Tensor, location: str):
    sketch = sketch.detach().cpu()
    digital = digital.detach().cpu()
    if sketch.ndim == 3:
        sketch = sketch.unsqueeze(0)
    if digital.ndim == 3:
        digital = digital.unsqueeze(0)
    concatenated = torch.cat([sketch, digital], dim=0)
    assert concatenated.ndim == 4
    grid = make_grid(
        concatenated, nrow=2, normalize=True, pad_value=1, padding=3
    )
    save_image(grid, location)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
