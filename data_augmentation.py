from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module
from torchvision import transforms
import albumentations as A
from PIL import Image
import numpy as np

import cv2
from pretrain_argparser import PretrainArgparser


class DataAugmentationV6(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        is_padding=True,
    ):
        super(DataAugmentationV6, self).__init__()
        self.image_size = image_size
        self.is_padding = is_padding

        self.to_tensor = transforms.ToTensor()

        # For rgb+depth+gt
        self.transform1 = A.Compose(
            [
                # A.CLAHE(),
                # A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.50,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                    mask_value=0,
                ),
                # A.Perspective(
                #     p=0.5,
                #     scale=(0.05, 0.2),
                #     pad_mode=cv2.BORDER_CONSTANT,
                #     pad_val=255,
                # ),
            ],
            additional_targets={"depth": "image", "gt": "mask"},
        )

        # For rgb only
        self.transform2 = A.Compose(
            [
                A.GaussianBlur(p=0.5, blur_limit=(3, 19)),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.5),
            ]
        )

        # For depth only
        self.transform3 = A.Compose([A.GaussianBlur(p=0.5, blur_limit=(3, 37))])

        # For rgb+depth+gt
        self.transform4 = A.Compose(
            [A.Resize(self.image_size, self.image_size)],
            additional_targets={"depth": "image", "gt": "mask"},
        )

        # For rgb only
        self.transform5 = A.Compose([A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # For depth only
        self.transform6 = A.Compose([A.Normalize(0.5, 0.5)])

        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.debug_idx = 0

    def forward(
        self,
        image: Image.Image,
        depth: Image.Image,
        is_transform: bool = True,  # is augmented?
        is_debug: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        if is_debug:
            import os

            # os.makedirs(f"tmp/{self.debug_idx}", exist_ok=True)
            # image.save(f"tmp/{self.debug_idx}/original_rgb.png")
            # depth.save(f"tmp/{self.debug_idx}/original_depth.png")
            # gt.save(f"tmp/{self.debug_idx}/original_gt.png")

        if not is_transform:
            # Dev or Test
            image, depth = self.resize_images([image, depth])
            image = np.array(image)
            image = self.transform5(image=image)["image"]
            # depth = self.transform6(image=depth)["image"]
            return self.to_tensors([image, depth])

        image = np.array(image)
        depth = np.array(depth)

        image = self.transform2(image=image)["image"]
        depth = self.transform3(image=depth)["image"]

        transformed = self.transform1(image=image, depth=depth)
        image = transformed["image"]
        depth = transformed["depth"]

        transformed = self.transform4(image=image, depth=depth)
        image = transformed["image"]
        depth = transformed["depth"]

        if is_debug:
            unnormalized_image = image
            # Image.fromarray(image).save(f"tmp/{self.debug_idx}/rgb.png")
            # Image.fromarray(depth).save(f"tmp/{self.debug_idx}/depth.png")
            # Image.fromarray(gt).save(f"tmp/{self.debug_idx}/gt.png")
            self.debug_idx += 1
            image = self.transform5(image=image)["image"]
            # depth = self.transform6(image=depth)["image"]
            return self.to_tensors([image, depth]) + [unnormalized_image]

        image = self.transform5(image=image)["image"]
        # depth = self.transform6(image=depth)["image"]
        return self.to_tensors([image, depth])

    def resize_images(self, lst: List[Image.Image]) -> List[Image.Image]:
        return [self.resize(e) for e in lst]

    def to_tensors(self, lst: List[Tensor]) -> List[Tensor]:
        return [self.to_tensor(e) for e in lst]