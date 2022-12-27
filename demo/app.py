import time
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torch import Tensor

from multimae.multimae import MultiMAE
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

def save_tensor_image(file_path: str, img: Tensor):
    if len(img.shape) == 2:
        img = torchvision.transforms.ToPILImage(mode='L')(
            img.unsqueeze(0)
        )
    else:
        img = torchvision.transforms.ToPILImage(mode='RGB')(
            img.permute(2,0,1)
        )
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, img)

def plot_predictions(
    input_dict: Dict[str, Tensor], preds: Dict[str, Tensor], 
    masks: Dict[str, Tensor], image_size=224,
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

    pred_rgb = get_pred_with_input(
        denormalize(input_dict['rgb']),
        denormalize(preds['rgb']).clamp(0,1),
        masks['rgb'],
        image_size=image_size
    )[0].permute(1,2,0).detach().cpu()
    pred_depth = get_pred_with_input(
        input_dict['depth'],
        preds['depth'],
        masks['depth'],
        image_size=image_size
    )[0,0].detach().cpu()

    save_tensor_image('masked_rgb.png', masked_rgb)
    save_tensor_image('pred_rgb.png', pred_rgb)
    save_tensor_image('rgb.png', denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu())
    
    save_tensor_image('masked_depth.png', masked_depth)
    save_tensor_image('pred_depth.png', pred_depth)
    save_tensor_image('depth.png', input_dict['depth'].squeeze().detach().cpu())
    
    print('ww')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(
    model: MultiMAE,
    input_dict: Dict[str, Tensor],
    num_tokens: int, 
    manual_mode: bool, 
    num_rgb: int, 
    num_depth: int, 
    seed: int,
):
    num_tokens = int(588 * num_tokens / 100.0)
    num_rgb = int(196 * num_rgb / 100.0)
    num_depth = int(196 * num_depth / 100.0)

    # To GPU
    input_dict = {k: v.to(device) for k,v in input_dict.items()}

    if not manual_mode:
        # Randomly sample masks
        torch.manual_seed(int(time.time())) # Random mode is random
        preds, masks = model.forward(
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
        task_masks['rgb'][:,selected_rgb_idxs] = 0
        task_masks['depth'][:,selected_depth_idxs] = 0

        preds, masks = model.forward(
            input_dict,
            mask_inputs=True,
            task_masks=task_masks
        )

    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    plot_predictions(input_dict, preds, masks)

    return 'output.png'
