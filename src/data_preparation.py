from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CycleDataset(Dataset):
    def __init__(self, root_x: str, root_y: str, transform=None):
        self.root_x = Path(root_x)
        self.root_y = Path(root_y)
        self.transform = transform

        self.x_images = [
            sample for sample in self.root_x.iterdir() if sample.is_file()
        ]
        self.y_images = [
            sample for sample in self.root_y.iterdir() if sample.is_file()
        ]

        self.dataset_length = max(len(self.x_images), len(self.y_images))
        self.x_len = len(self.x_images)
        self.y_len = len(self.y_images)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        x_path = self.x_images[idx % self.x_len]
        y_path = self.y_images[idx % self.y_len]
        x_img = np.array(Image.open(x_path).convert("RGB"))
        y_img = np.array(Image.open(y_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=x_img, image0=y_img)
            x_img = augmentations["image"]
            y_img = augmentations["image0"]

        return x_img, y_img


if __name__ == "__main__":
    dataset = CycleDataset("X-Images", "Y-Images")
    for a, b in dataset:
        plt.imshow(a)
        plt.show()
        plt.imshow(b)
        plt.show()
        break
