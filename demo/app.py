import time
from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torch import Tensor

from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_masked_image(
    img: Tensor, mask: Tensor, image_size=224, 
    patch_size=16, mask_value=0.0,
) -> Tensor:
    img_token = rearrange(
        img.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    img_token[mask.detach().cpu()!=0] = mask_value
    img = rearrange(
        img_token,
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img


def denormalize(
    img: Tensor, 
    mean=IMAGENET_DEFAULT_MEAN, 
    std=IMAGENET_DEFAULT_STD,
) -> Tensor:
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )
    
def get_pred_with_input(
    gt: Tensor, pred: Tensor, mask: Tensor, 
    image_size=224, patch_size=16,
) -> Tensor:
    gt_token = rearrange(
        gt.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token = rearrange(
        pred.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token[mask.detach().cpu()==0] = gt_token[mask.detach().cpu()==0]
    img = rearrange(
        pred_token,
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img    

def plot_predictions(
    input_dict: Dict[str, Any], preds: Dict[str, Any], 
    masks: Dict[str, Any], image_size=224,
):
    masked_rgb = get_masked_image(
        denormalize(input_dict['rgb']),
        masks['rgb'],
        image_size=image_size,
        mask_value=1.0
    )[0].permute(1,2,0).detach().cpu()
    masked_depth = get_masked_image(
        input_dict['depth'],
        masks['depth'],
        image_size=image_size,
        mask_value=np.nan
    )[0,0].detach().cpu()

    pred_rgb2 = get_pred_with_input(
        denormalize(input_dict['rgb']),
        denormalize(preds['rgb']).clamp(0,1),
        masks['rgb'],
        image_size=image_size
    )[0].permute(1,2,0).detach().cpu()
    pred_depth2 = get_pred_with_input(
        input_dict['depth'],
        preds['depth'],
        masks['depth'],
        image_size=image_size
    )[0,0].detach().cpu()

    masked_rgb
    pred_rgb2
    denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu()

    masked_depth
    pred_depth2
    input_dict['depth'][0,0].detach().cpu()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(
    img: str, 
    num_tokens: int, 
    manual_mode: bool, 
    num_rgb: int, 
    num_depth: int, 
    seed: int,
):
    num_tokens = int(588 * num_tokens / 100.0)
    num_rgb = int(196 * num_rgb / 100.0)
    num_depth = int(196 * num_depth / 100.0)
    num_semseg = int(196 * num_semseg / 100.0)

    im = Image.open(img)

    # Center crop and resize RGB
    image_size = 224 # Train resolution
    img = TF.center_crop(TF.to_tensor(im), min(im.size))
    img = TF.resize(img, image_size, interpolation=TF.InterpolationMode.BICUBIC)

    # Pre-process RGB, depth and semseg to the MultiMAE input format
    input_dict = {}

    # Normalize RGB
    input_dict['rgb'] = TF.normalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD).unsqueeze(0)

    # Normalize depth robustly
    trunc_depth = torch.sort(depth.flatten())[0]
    trunc_depth = trunc_depth[int(0.1 * trunc_depth.shape[0]): int(0.9 * trunc_depth.shape[0])]
    depth = (depth - trunc_depth.mean()[None,None,None]) / torch.sqrt(trunc_depth.var()[None,None,None] + 1e-6)
    input_dict['depth'] = depth.unsqueeze(0)

    # Downsample semantic segmentation
    stride = 4

    # To GPU
    input_dict = {k: v.to(device) for k,v in input_dict.items()}

    if not manual_mode:
        # Randomly sample masks

        torch.manual_seed(int(time.time())) # Random mode is random

        preds, masks = multimae.forward(
            input_dict,
            mask_inputs=True, # True if forward pass should sample random masks
            num_encoded_tokens=num_tokens,
            alphas=1.0
        )
    else:
        # Randomly sample masks using the specified number of tokens per modality

        torch.manual_seed(int(seed)) # change seed to resample new mask

        task_masks = {domain: torch.ones(1,196).long().to(device) for domain in DOMAINS}
        selected_rgb_idxs = torch.randperm(196)[:num_rgb]
        selected_depth_idxs = torch.randperm(196)[:num_depth]
        selected_semseg_idxs = torch.randperm(196)[:num_semseg]
        task_masks['rgb'][:,selected_rgb_idxs] = 0
        task_masks['depth'][:,selected_depth_idxs] = 0
        task_masks['semseg'][:,selected_semseg_idxs] = 0

        preds, masks = multimae.forward(
            input_dict,
            mask_inputs=True,
            task_masks=task_masks
        )

    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    plot_predictions(input_dict, preds, masks)

    return 'output.png'
