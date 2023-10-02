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
