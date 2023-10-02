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


class Discriminator(nn.Module):
    """
    The discriminator that will be used in the CycleGAN
    Since the CycleGAN's discriminator is a patchGAN we will be outputting a patch of size (30, 30)
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: list = [64, 128, 256, 512],
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self.init_type = init_type
        self.init_gain = init_gain

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

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        classname = m.__class__.__name__

        init_type = self.init_type
        init_gain = self.init_gain

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                torch.nn.init.normal_(
                    tensor=m.weight.data, mean=0.0, std=init_gain
                )
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(
                    tensor=m.weight.data, gain=init_gain
                )
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(
                    tensor=m.weight.data, a=0, mode="fan_in"
                )
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(tensor=m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(tensor=m.bias.data, val=0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(
                tensor=m.weight.data, mean=1.0, std=init_gain
            )
            torch.nn.init.constant_(tensor=m.bias.data, val=0.0)

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
