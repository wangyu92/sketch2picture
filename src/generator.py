import torch
import torch.nn as nn
from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_sample: bool = True,
        use_act: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding_mode="reflect",
                **kwargs
            )
            if down_sample
            else nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, **kwargs
            ),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),  # Same Convolution
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                use_act=False,
                kernel_size=3,
                padding=1,
            )
            # Same Convolution with no activation
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    A generator that performs image-to-image translation.
    Takes an image of size (3, 256, 256) returns an image of size (3, 256, 256)
    """

    # n_residuals = 9 for (256, 256) and 6 for (128, 128)
    def __init__(
        self,
        img_channels: int,
        n_features: int = 64,
        n_residuals: int = 9,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self.init_type = init_type
        self.init_gain = init_gain

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=n_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features=n_features),
            nn.ReLU(inplace=True),
        )
        self.downsample_blocks = nn.ModuleList(
            modules=[
                ConvBlock(
                    in_channels=n_features,
                    out_channels=n_features * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ConvBlock(
                    in_channels=n_features * 2,
                    out_channels=n_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(channels=n_features * 4)
                for _ in range(n_residuals)
            ]
        )
        self.upsample_blocks = nn.ModuleList(
            modules=[
                ConvBlock(
                    in_channels=n_features * 4,
                    out_channels=n_features * 2,
                    down_sample=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    in_channels=n_features * 2,
                    out_channels=n_features,
                    down_sample=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )
        # The final layer that outputs an image
        self.last = nn.Sequential(
            nn.Conv2d(
                in_channels=n_features,
                out_channels=img_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )

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
        for layer in self.downsample_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.upsample_blocks:
            x = layer(x)
        return self.last(x)


if __name__ == "__main__":
    img_channels = 3
    size = 256
    inputs = torch.randn((1, img_channels, size, size))
    generator = Generator(img_channels, 9)
    image = generator(inputs)
    assert image.shape[1:] == torch.Size([3, 256, 256])
    print("Success")

    summary(generator, input_size=(1, 3, 256, 256))
