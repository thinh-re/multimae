# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on BEiT, timm, DINO, DeiT and MAE-priv code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
import torchvision.transforms.functional as transformsF
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import albumentations

from pretrain_argparser import PretrainArgparser
from utils import create_transform

from .data_constants import (
    IMAGE_TASKS,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from .dataset_folder import ImageFolder, MultiTaskImageFolder, MultiTaskImageFolderV2


def denormalize(
    img: Tensor,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> Tensor:
    return TF.normalize(
        img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


class DataAugmentationForMAE(object):
    def __init__(self, args: PretrainArgparser):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = (
            IMAGENET_INCEPTION_MEAN
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_MEAN
        )
        std = (
            IMAGENET_INCEPTION_STD
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_STD
        )

        trans = [transforms.RandomResizedCrop(args.input_size)]
        if args.hflip > 0.0:
            trans.append(transforms.RandomHorizontalFlip(args.hflip))
        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

        self.transform = transforms.Compose(trans)

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


class DataAugmentationForMultiMAE(object):
    def __init__(
        self,
        args: PretrainArgparser,
        eval_mode: bool = False,
    ):
        self.eval_mode = eval_mode
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        self.rgb_mean = (
            IMAGENET_INCEPTION_MEAN
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_MEAN
        )
        self.rgb_std = (
            IMAGENET_INCEPTION_STD
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_STD
        )
        self.input_size = args.input_size
        self.hflip = args.hflip
        self.depth_range = args.depth_range
        self.du = None
        if args.data_augmentation_version == 2:
            self.du = DataAugmentationV2(self.input_size, ["depth", "rgb"], ["depth", "rgb"])

    def __call__(self, task_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self.eval_mode:
            flip = (
                random.random() < self.hflip
            )  # Stores whether to flip all images or not
            ijhw = None  # Stores crop coordinates used for all tasks

            # Crop and flip all tasks randomly, but consistently for all tasks
            for task in task_dict:
                if task not in IMAGE_TASKS:
                    continue
                if ijhw is None:
                    # Official MAE code uses (0.2, 1.0) for scale and (0.75, 1.3333) for ratio
                    ijhw = transforms.RandomResizedCrop.get_params(
                        task_dict[task], scale=(0.2, 1.0), ratio=(0.75, 1.3333)
                    )
                i, j, h, w = ijhw
                task_dict[task] = TF.crop(task_dict[task], i, j, h, w)
                task_dict[task] = task_dict[task].resize(
                    (self.input_size, self.input_size)
                )
                if flip:
                    task_dict[task] = TF.hflip(task_dict[task])

        if self.du is None:
            for task in task_dict:
                if task in ["depth"]:
                    if self.eval_mode:
                        img = TF.to_tensor(task_dict[task])
                        img = TF.center_crop(img, min(img.shape[1:]))
                        img = TF.resize(
                            img,
                            self.input_size,
                            interpolation=TF.InterpolationMode.BICUBIC,
                        )
                    else:
                        img = torch.Tensor(np.array(task_dict[task]) / self.depth_range)
                        img = img.unsqueeze(0)  # 1 x H x W
                elif task in ["rgb"]:
                    img = TF.to_tensor(task_dict[task])
                    if self.eval_mode:
                        img = TF.center_crop(img, min(img.shape[1:]))
                        img = TF.resize(
                            img,
                            self.input_size,
                            interpolation=TF.InterpolationMode.BICUBIC,
                        )
                    img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)

                task_dict[task] = img
        else:
            if self.eval_mode:
                for task in task_dict:
                    if task in ["depth"]:
                        img = TF.to_tensor(task_dict[task])
                        img = TF.center_crop(img, min(img.shape[1:]))
                        img = TF.resize(
                            img,
                            self.input_size,
                            interpolation=TF.InterpolationMode.BICUBIC,
                        )
                    elif task in ["rgb"]:
                        img = TF.to_tensor(task_dict[task])
                        img = TF.center_crop(img, min(img.shape[1:]))
                        img = TF.resize(
                            img,
                            self.input_size,
                            interpolation=TF.InterpolationMode.BICUBIC,
                        )
                        img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)

                    task_dict[task] = img
            else:
                assert "depth" in task_dict and "rgb" in task_dict
                depth = task_dict["depth"]
                rgb = task_dict["rgb"]
                task_dict["rgb"], task_dict["depth"], _ = self.du.forward(rgb, depth)

        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def random_choice(p: float) -> bool:
    """Return True if random float <= p"""
    return random.random() <= p


class SquarePad:
    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value

    def __call__(self, image: Image) -> Image:
        _, w, h = image.shape
        max_wh = np.max([w, h])
        wp = int((max_wh - w) / 2)
        hp = int((max_wh - h) / 2)
        padding = (hp, hp, wp, wp)
        image = F.pad(image, padding, value=self.fill_value, mode="constant")
        return image


class DataAugmentationV2(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        inputs: List[str],
        outputs: List[str],
        is_padding=True,
    ):
        super(DataAugmentationV2, self).__init__()
        self.image_size = image_size
        self.is_padding = is_padding
        self.inputs = inputs
        self.outputs = outputs

        self.to_tensor = transforms.ToTensor()
        self.to_image = transforms.ToPILImage()
        self.square_pad_0 = SquarePad(fill_value=0.0)  # for rgb, gt
        self.square_pad_1 = SquarePad(fill_value=1.0)  # for depth
        self.resize = transforms.Resize((self.image_size, self.image_size))

        self.random_perspective_0 = transforms.RandomPerspective(
            distortion_scale=0.2, p=1.0, fill=0.0
        )
        self.random_perspective_1 = transforms.RandomPerspective(
            distortion_scale=0.2, p=1.0, fill=255
        )

        self.longest_max_size = (
            albumentations.augmentations.geometric.resize.LongestMaxSize(
                max_size=self.image_size, p=1
            )
        )

        # RGB, p = 0.5
        self.transform_color_jitter = transforms.ColorJitter(brightness=0.5, hue=0.3)

        # RGB, p = 1.0
        self.transform_contrast_sharpness = transforms.Compose(
            [
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            ]
        )

        self.normalize_image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def no_pad_resize(self, lst: List[Tensor]) -> List[Tensor]:
        return [self.resize(e) for e in lst]

    def pad_resize(self, lst: List[Tensor]) -> List[Tensor]:
        gt: Tensor = None
        if len(lst) == 3:
            image, depth, gt = lst
        else:
            image, depth = lst

        image = self.to_tensor(image)
        image = self.square_pad_0(image)
        image = self.resize(image)
        image = self.to_image(image)

        if gt is not None:
            gt = self.to_tensor(gt)
            gt = self.square_pad_0(gt)
            gt = self.resize(gt)
            gt = self.to_image(gt)

        depth = self.to_tensor(depth)
        depth = self.square_pad_1(depth)
        depth = self.resize(depth)
        depth = self.to_image(depth)

        if gt is not None:
            return [image, depth, gt]
        else:
            return [image, depth]

    def process_transform_to_tensor(self, lst: List[Tensor]) -> List[Tensor]:
        return [self.to_tensor(e) for e in lst]

    def random_horizontal_flip(self, lst: List[Tensor], p=0.5) -> List[Tensor]:
        if random_choice(p=p):
            return [transformsF.hflip(e) for e in lst]
        return lst

    def random_vertical_flip(self, lst: List[Tensor], p=0.5) -> List[Tensor]:
        if random_choice(p=p):
            return [transformsF.vflip(e) for e in lst]
        return lst

    def random_rotate(self, lst: List[Tensor], p=0.3) -> List[Tensor]:
        if random_choice(p=p):
            angle = transforms.RandomRotation.get_params(degrees=(0, 90))

            rs: List[Tensor] = []
            for i, e in enumerate(lst):
                if i == 1:
                    rs.append(
                        transformsF.rotate(
                            e, angle, InterpolationMode.BICUBIC, fill=255
                        )
                    )
                else:
                    rs.append(transformsF.rotate(e, angle, InterpolationMode.BICUBIC))
            return rs
        return lst

    def random_resized_crop(self, lst: List[Tensor], p=0.3) -> List[Tensor]:
        if random_choice(p=p):
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                lst[0], scale=(0.5, 2.0), ratio=(0.75, 1.3333333333333333)
            )
            return [
                transformsF.resized_crop(
                    e,
                    i,
                    j,
                    h,
                    w,
                    [self.image_size, self.image_size],
                    InterpolationMode.BICUBIC,
                )
                for e in lst
            ]
        return lst

    def random_gaussian_blur(
        self,
        tensor: Tensor,
        p=0.5,
        max_kernel_size: int = 19,  # must be an odd positive integer
    ) -> Tensor:
        if random_choice(p=p):
            kernel_size = random.randrange(1, max_kernel_size, 2)
            return transformsF.gaussian_blur(tensor, kernel_size=kernel_size)
        return tensor

    def color_jitter(self, tensor: Tensor, p=0.5) -> Tensor:
        if random_choice(p=p):
            return self.transform_color_jitter(tensor)
        return tensor

    def random_maskout_depth(self, tensor: Tensor, p=0.5) -> Tensor:
        if random_choice(p=p):
            _, h, w = tensor.shape
            xs = np.random.choice(w, 2)
            ys = np.random.choice(h, 2)
            tensor[:, min(ys) : max(ys), min(xs) : max(xs)] = torch.ones(
                (max(ys) - min(ys), max(xs) - min(xs))
            )
            return tensor
        return tensor

    def random_perspective(self, lst: List[Tensor], p=0.2) -> List[Tensor]:
        if random_choice(p=p):
            gt: Tensor = None
            if len(lst) == 3:
                image, depth, gt = lst
            else:
                image, depth = lst

            image = self.random_perspective_0(image)

            if gt is not None:
                gt = self.random_perspective_0(gt)

            depth = self.random_perspective_1(depth)

            if gt is not None:
                return [image, depth, gt]
            else:
                return [image, depth]
        return lst

    def preprocessing(self, images: Tensor, depths: Tensor) -> Tuple[Tensor, Tensor]:
        images, depths = self.resize(images), self.resize(depths)
        return self.normalize_image(images), depths

    def forward(
        self,
        image: Image.Image,
        depth: Image.Image,
        gt: Optional[Image.Image] = None,
        is_transform: Optional[bool] = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        lst = [image, depth, gt] if gt is not None else [image, depth]

        if not is_transform:
            # Dev or Test
            if self.is_padding:
                lst = self.pad_resize(lst)
            else:
                lst = self.no_pad_resize(lst)
            lst = self.process_transform_to_tensor(lst)
            if gt is not None:
                image, depth, gt = lst
                # gt[gt > 0.0] = 1.0
            else:
                image, depth = lst
            image = self.normalize_image(image)
            return image, depth, gt

        lst = self.random_horizontal_flip(lst)
        if random_choice(p=0.2):
            lst = self.pad_resize(lst)
        else:
            lst = self.no_pad_resize(lst)
        lst = self.random_perspective(lst, p=0.2)
        lst = self.random_rotate(lst)
        lst = self.random_resized_crop(lst)
        lst = self.process_transform_to_tensor(lst)

        if gt is not None:
            image, depth, gt = lst
        else:
            image, depth = lst

        image = self.color_jitter(image)
        image = self.transform_contrast_sharpness(image)
        image = self.random_gaussian_blur(image, p=0.5, max_kernel_size=19)
        if "depth" in self.inputs:
            depth = self.random_gaussian_blur(depth, p=0.5, max_kernel_size=36)
        image = self.normalize_image(image)

        return image, depth, gt


def build_pretraining_dataset(args: PretrainArgparser):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)


def build_multimae_pretraining_train_dataset_v2(args: PretrainArgparser):
    transform = DataAugmentationForMultiMAE(args)
    return MultiTaskImageFolderV2(
        args.data_paths,
        args.all_domains,
        transform=transform,
        normalized_depth=args.normalized_depth,
    )


# @deprecated
def build_multimae_pretraining_train_dataset(args: PretrainArgparser):
    transform = DataAugmentationForMultiMAE(args)
    return MultiTaskImageFolder(args.data_path, args.all_domains, transform=transform)


def build_multimae_pretraining_dev_dataset_v2(args: PretrainArgparser):
    transform = DataAugmentationForMultiMAE(args, eval_mode=True)
    return MultiTaskImageFolderV2(
        args.data_paths,
        args.all_domains,
        transform=transform,
        normalized_depth=args.normalized_depth,
    )


# @deprecated
def build_multimae_pretraining_dev_dataset(args: PretrainArgparser):
    transform = DataAugmentationForMultiMAE(args, eval_mode=True)
    return MultiTaskImageFolder(args.data_path, args.all_domains, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == "CIFAR":
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == "IMNET":
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (
        IMAGENET_INCEPTION_MEAN
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_MEAN
    )
    std = (
        IMAGENET_INCEPTION_STD
        if not imagenet_default_mean_and_std
        else IMAGENET_DEFAULT_STD
    )

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
