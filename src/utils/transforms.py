"""Image transforms for the FF++ EfficientNet-B0 detector."""

from __future__ import annotations

from torchvision import transforms

from config import IMAGE_SIZE


def get_image_transform() -> transforms.Compose:
    """Matches FaceForensics++ C23 training (resize + tensor, no ImageNet norm)."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
