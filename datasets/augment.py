"""
Function:
Randomly augment images and their corresponding segmentation masks.
The script applies the same geometric transforms to both images and masks,
then saves the augmented pairs with unique filenames.
"""

import os
import cv2
import random
import albumentations as A
from tqdm import tqdm
from datetime import datetime


# =========================
# Settings
# =========================

IMAGE_DIR = "../myDataset/data/leftImg8bit/train/aachen"
MASK_DIR = "../myDataset/data/gtFine/train/aachen"

# Number of augmentations
AUG_NUM = 14000


# =========================
# Augmentation Pipeline
# =========================

transform = A.Compose(
    [
        # Geometry
        A.Rotate(
            limit=30,
            p=0.5
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.VerticalFlip(
            p=0.3
        ),

        # Crop and resize
        A.RandomResizedCrop(
            size=(1024, 1024),
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),

        # Lighting
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),

        A.RandomGamma(
            gamma_limit=(70, 130),
            p=0.3
        ),

        # Weather effects
        A.RandomShadow(
            p=0.2
        ),

        # Blur
        A.OneOf(
            [
                A.GaussianBlur(
                    blur_limit=3
                ),
                A.MotionBlur(
                    blur_limit=5
                )
            ],
            p=0.2
        ),

        # Noise
        A.GaussNoise(
            p=0.2
        ),
    ]
)


# =========================
# Main
# =========================

def main():

    # Get original images
    images = [
        f for f in os.listdir(IMAGE_DIR)
        if f.endswith(".png")
        and "re" not in f
    ]

    print(f"Original images: {len(images)}")

    # Check images
    if len(images) == 0:
        raise ValueError("No original images found!")

    print(f"Starting {AUG_NUM} augmentations")
    print("Images can be selected multiple times")

    # Random selection with replacement
    selected = random.choices(
        images,
        k=AUG_NUM
    )

    # Start augmentation
    for index, img_name in enumerate(
        tqdm(selected),
        start=1
    ):

        # Image path
        img_path = os.path.join(
            IMAGE_DIR,
            img_name
        )

        # Mask path
        mask_path = os.path.join(
            MASK_DIR,
            img_name
        )

        # Check mask
        if not os.path.exists(mask_path):
            print(f"Mask not found: {img_name}")
            continue

        # Read image and mask
        image = cv2.imread(
            img_path
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Check loading
        if image is None:
            print(f"Failed to read image: {img_name}")
            continue

        if mask is None:
            print(f"Failed to read mask: {img_name}")
            continue

        # Apply synchronized augmentation
        augmented = transform(
            image=image,
            mask=mask
        )

        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        # Create unique filename
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        base_name, ext = os.path.splitext(
            img_name
        )

        # Original name + timestamp + index + suffix
        name = (
            f"{base_name}_"
            f"{timestamp}_"
            f"{index}"
            f"re{ext}"
        )

        # Save augmented image
        cv2.imwrite(
            os.path.join(
                IMAGE_DIR,
                name
            ),
            aug_img
        )

        # Save augmented mask
        cv2.imwrite(
            os.path.join(
                MASK_DIR,
                name
            ),
            aug_mask
        )

    print(f"Augmentation completed: {AUG_NUM} images generated")


if __name__ == "__main__":
    main()