from torchvision.transforms import ToPILImage
import time
from typing import Dict, Optional, Tuple

from PIL import Image
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from torch import Tensor, nn

import wandb
from multimae.multimae import MultiMAE
from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.dataset_folder import MultiTaskImageFolderV2
from utils.logger import WandbLogger


def get_masked_image(
    img: Tensor,
    mask: Tensor,
    image_size=224,
    patch_size=16,
    mask_value=0.0,
) -> Tensor:
    img_token = rearrange(
        img.detach().cpu(),
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    img_token[mask.detach().cpu() != 0] = mask_value
    img = rearrange(
        img_token,
        "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    return img


def denormalize(
    img: Tensor,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
) -> Tensor:
    return TF.normalize(
        img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def get_pred_with_input(
    gt: Tensor,
    pred: Tensor,
    mask: Tensor,
    image_size=224,
    patch_size=16,
) -> Tensor:
    gt_token = rearrange(
        gt.detach().cpu(),
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    pred_token = rearrange(
        pred.detach().cpu(),
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    pred_token[mask.detach().cpu() == 0] = gt_token[mask.detach().cpu() == 0]
    img = rearrange(
        pred_token,
        "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    return img


def to_cv_image(img: Tensor) -> np.ndarray:
    if len(img.shape) == 2:
        img = torchvision.transforms.ToPILImage(mode="L")(img.unsqueeze(0))
    else:
        img = torchvision.transforms.ToPILImage(mode="RGB")(img.permute(2, 0, 1))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


toPILImage = ToPILImage()


def generate_predictions(
    input_dict: Dict[str, Tensor],
    preds: Dict[str, Tensor],
    masks: Dict[str, Tensor],
    image_size=224,
) -> Tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
]:
    masked_rgb = (
        get_masked_image(
            denormalize(input_dict["rgb"]),
            masks["rgb"],
            image_size=image_size,
            mask_value=1.0,
        )[0]
        .detach()
        .cpu()
    )
    masked_depth = (
        get_masked_image(
            input_dict["depth"],
            masks["depth"],
            image_size=image_size,
            mask_value=np.nan,
        )[0, 0]
        .detach()
        .cpu()
    )

    pred_rgb = (
        get_pred_with_input(
            denormalize(input_dict["rgb"]),
            denormalize(preds["rgb"]).clamp(0, 1),
            masks["rgb"],
            image_size=image_size,
        )[0]
        .detach()
        .cpu()
    )
    pred_depth = (
        get_pred_with_input(
            input_dict["depth"], preds["depth"], masks["depth"], image_size=image_size
        )[0, 0]
        .detach()
        .cpu()
    )

    return (
        toPILImage(masked_rgb),
        toPILImage(pred_rgb),
        toPILImage(denormalize(input_dict["rgb"])[0].detach().cpu()),
        toPILImage(masked_depth),
        toPILImage(pred_depth),
        toPILImage(input_dict["depth"].squeeze().detach().cpu()),
    )


device = "cuda" if torch.cuda.is_available() else "cpu"

l1loss = nn.L1Loss()


def log_inference(
    model: MultiMAE,
    seed: int,
    dataset_dev: MultiTaskImageFolderV2,
    log_writer: Optional[WandbLogger],
    epoch: int,
    num_samples: int = 10,
):
    if log_writer is None:
        return
    print("log_inference")
    data = [
        # [wandb.Image, wandb.Image, wandb.Image, wandb.Image, wandb.Image, wandb.Image, ]
    ]
    for i in range(num_samples):
        dev_input_dict = dataset_dev.__getitem__(i)[0]
        masked_rgb, pred_rgb, rgb, masked_depth, pred_depth, depth = inference(
            model,
            dev_input_dict,
            num_tokens=15,
            manual_mode=False,
            num_rgb=15,
            num_depth=15,
            seed=seed,
        )
        rgb_mae = l1loss(Tensor(rgb), Tensor(pred_rgb))
        depth_mae = l1loss(Tensor(depth), Tensor(pred_depth))
        data.append(
            [
                wandb.Image(masked_rgb),
                wandb.Image(pred_rgb),
                wandb.Image(rgb),
                rgb_mae,
                wandb.Image(masked_depth),
                wandb.Image(pred_depth),
                wandb.Image(depth),
                depth_mae,
            ]
        )
    log_writer.update(
        {
            f"inference/{epoch}": wandb.Table(
                data=data,
                columns=[
                    "masked_rgb",
                    "pred_rgb",
                    "rgb",
                    "rgb_mae",
                    "masked_depth",
                    "pred_depth",
                    "depth",
                    "depth_mae",
                ],
            )
        }
    )


def inference(
    model: MultiMAE,
    input_dict: Dict[str, Tensor],
    num_tokens: int,
    num_rgb: int,
    num_depth: int,
    image_size: int,
):
    num_tokens = int(196 * 2 * num_tokens / 100.0)  # should 2
    num_rgb = int(196 * num_rgb / 100.0)
    num_depth = int(196 * num_depth / 100.0)

    # To GPU
    input_dict = {k: v.unsqueeze(0).to(model.device) for k, v in input_dict.items()}
    # Randomly sample masks
    torch.manual_seed(int(time.time()))  # Random mode is random
    preds, masks = model.forward(
        input_dict,
        mask_inputs=True,  # True if forward pass should sample random masks
        num_encoded_tokens=num_tokens,
        alphas=1.0,
    )

    preds: Dict[str, Tensor]
    masks: Dict[str, Tensor]
    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    return generate_predictions(input_dict, preds, masks, image_size=image_size)
