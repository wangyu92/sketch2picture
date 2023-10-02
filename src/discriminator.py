import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),  # KernelSize is always 4 according to the implementation
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(
    nn.Module
):  # The discriminator that will be used in the CycleGAN
    # Since the CycleGAN's discriminator is a patchGAN we will be outputting a patch of size (30, 30)
    def __init__(
        self, in_channels: int = 3, features: list = [64, 128, 256, 512]
    ):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            # Except for the last layer all layers have a stride of 2
            layers.append(
                Block(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature

        # Make the channels as 1 aka the output Patch
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.Sigmoid(),
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return self.model(x)


if __name__ == "__main__":
    # A sample test case to see if the discriminator model works fine

    inputs = torch.randn((2, 3, 256, 256))
    disc = Discriminator()
    preds = disc(inputs)
    assert preds.shape[1:] == torch.Size([1, 30, 30])
    print("Success")
