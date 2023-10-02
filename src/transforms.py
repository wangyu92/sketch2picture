import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255
        ),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False,
)


def get_transforms(load_size: int, crop_size: int, is_train: bool = True):
    if is_train:
        return A.Compose(
            [
                A.Resize(width=load_size, height=load_size),
                A.RandomCrop(width=crop_size, height=crop_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
            is_check_shapes=False,
        )
    else:
        return A.Compose(
            [
                A.Resize(width=crop_size, height=crop_size),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
            is_check_shapes=False,
        )
