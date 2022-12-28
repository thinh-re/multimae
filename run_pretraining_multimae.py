# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------
import datetime
import json
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from einops import rearrange
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from demo.app import log_inference

import utils
from multimae import MultiMAE
from multimae.criterion import (MaskedCrossEntropyLoss, MaskedL1Loss,
                                MaskedMSELoss)
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from pretrain_argparser import PretrainArgparser, get_args
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import COCO_SEMSEG_NUM_CLASSES
from utils.datasets import (build_multimae_pretraining_dev_dataset,
                            build_multimae_pretraining_train_dataset)
from utils.optim_factory import create_optimizer

CPU_DEVICE = torch.device('cpu')

DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
        'loss': MaskedMSELoss,
    },
    'depth': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
        'loss': MaskedL1Loss,
    },
    'semseg': {
        'num_classes': 133,
        'stride_level': 4,
        'input_adapter': partial(
            SemSegInputAdapter, num_classes=COCO_SEMSEG_NUM_CLASSES,
            dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=COCO_SEMSEG_NUM_CLASSES),
        'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}

def get_model(args: PretrainArgparser) -> MultiMAE:
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )
        for domain in args.out_domains
    }

    # Add normalized pixel output adapter if specified
    if args.extra_norm_pix_loss:
        output_adapters['norm_rgb'] = DOMAIN_CONF['rgb']['output_adapter'](
            stride_level=DOMAIN_CONF['rgb']['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task='rgb',
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path
    )

    return model

def main(args: PretrainArgparser):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](
            patch_size=args.patch_size, 
            stride=DOMAIN_CONF[domain]['stride_level'])
        for domain in args.out_domains
    }

    # Add normalized pixel loss if specified
    if args.extra_norm_pix_loss:
        tasks_loss_fn['norm_rgb'] = DOMAIN_CONF['rgb']['loss'](
            patch_size=args.patch_size,
            stride=DOMAIN_CONF['rgb']['stride_level'],
            norm_pix=True)

    # Get dataset
    dataset_train = build_multimae_pretraining_train_dataset(args)
    dataset_dev = build_multimae_pretraining_dev_dataset(args)    

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, 
            rank=sampler_rank, shuffle=True, drop_last=True,
        )
        # sampler_dev = DistributedSampler(
        #     dataset_dev, num_replicas=num_tasks, 
        #     rank=sampler_rank, shuffle=False, drop_last=False,
        # )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = RandomSampler(dataset_train)

    print(args)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # data_loader_dev = DataLoader(
    #     dataset_dev, sampler=sampler_dev,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False,
    # )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, {'model': model_without_ddp})
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, 
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, 
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, 
        args.epochs, num_training_steps_per_epoch,
    )
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, 
        model_without_ddp=model_without_ddp, 
        optimizer=optimizer, loss_scaler=loss_scaler,
    )
    
    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
        log_writer.set_step(0)
    else:
        log_writer = None
            
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens=args.num_encoded_tokens,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
            standardize_depth=args.standardize_depth,
            extra_norm_pix_loss=args.extra_norm_pix_loss,
            fp32_output_adapters=args.fp32_output_adapters.split('-'),
        )
        if log_writer is not None:
            log_writer.update({**{k: v for k, v in train_stats.items()}, 'epoch': epoch})
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, 
                    model_without_ddp=model_without_ddp, 
                    optimizer=optimizer,
                    loss_scaler=loss_scaler, 
                    epoch=epoch,
                )
                
                log_inference(
                    model, seed, dataset_dev, 
                    log_writer, epoch='latest', num_samples=10,
                )

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        # if epoch % 5 == 0:
        #     eval(
        #         dataset_type = 'dev',
        #         args = args,
        #         model = model, 
        #         data_loader = data_loader_dev, 
        #         tasks_loss_fn = tasks_loss_fn,
        #         device = device, 
        #         epoch = epoch,
        #         log_writer = log_writer,
        #         num_encoded_tokens=args.num_encoded_tokens,
        #         in_domains=args.in_domains,
        #         loss_on_unmasked=args.loss_on_unmasked,
        #         alphas=args.alphas,
        #         sample_tasks_uniformly=args.sample_tasks_uniformly,
        #         standardize_depth=args.standardize_depth,
        #         extra_norm_pix_loss=args.extra_norm_pix_loss,
        #         fp32_output_adapters=args.fp32_output_adapters.split('-'),
        #     )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(
    model: DistributedDataParallel, 
    data_loader: DataLoader, 
    tasks_loss_fn: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    loss_scaler: NativeScaler, 
    max_norm: Optional[float] = None, 
    max_skip_norm: Optional[float] = None,
    log_writer: Optional[utils.WandbLogger]=None, 
    start_steps: Optional[int] = None, 
    lr_schedule_values: Optional[np.ndarray]=None, 
    wd_schedule_values: Optional[np.ndarray]=None,
    num_encoded_tokens: int = 196, 
    in_domains: List[str] = [], 
    loss_on_unmasked: bool = True,
    alphas: float = 1.0, 
    sample_tasks_uniformly: bool = False, 
    standardize_depth: bool = True,
    extra_norm_pix_loss: bool = False, 
    fp32_output_adapters: List[str] = [],
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        x: Dict[str, Tensor] = x
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        # Truncated depth standardization
        if standardize_depth and 'depth' in tasks_dict:
            # Flatten depth and remove bottom and top 10% of values
            trunc_depth = torch.sort(rearrange(tasks_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
            trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
            tasks_dict['depth'] = (tasks_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with torch.cuda.amp.autocast():
            results = model(
                input_dict, 
                num_encoded_tokens=num_encoded_tokens, 
                alphas=alphas, 
                sample_tasks_uniformly=sample_tasks_uniformly,
                fp32_output_adapters=fp32_output_adapters,
            )
            '''
            {
                'rgb': tensor of shape (b, c, h, w)
                'depth': tensor of shape (b, 1, h, w)
                'norm_rgb': tensor of shape (b, c, h, w)
            }
            '''
            preds: Dict[str, Tensor] = results[0]
            masks: Dict[str, Tensor] = results[1]

            if extra_norm_pix_loss:
                tasks_dict['norm_rgb'] = tasks_dict['rgb']
                masks['norm_rgb'] = masks.get('rgb', None)

            task_losses: Dict[str, Tensor] = {}
            for task in preds:
                target = tasks_dict[task]
                    
                if loss_on_unmasked:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                else:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))

            loss = sum(task_losses.values())

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
            parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.update(task_loss_values)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval(
    dataset_type: str, # 'dev', 'test'
    args: PretrainArgparser,
    model: DistributedDataParallel, 
    data_loader: DataLoader, 
    tasks_loss_fn: Dict[str, torch.nn.Module],
    device: torch.device,
    epoch: int,
    log_writer: Optional[utils.WandbLogger]=None,
    num_encoded_tokens: int = 196, 
    in_domains: List[str] = [], 
    loss_on_unmasked: bool = True,
    alphas: float = 1.0, 
    sample_tasks_uniformly: bool = False, 
    standardize_depth: bool = True,
    extra_norm_pix_loss: bool = False, 
    fp32_output_adapters: List[str] = [],
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    total_loss: float = 0.0
    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        x: Dict[str, Tensor] = x

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }
        
        batch_size = 0
        for tensor in x.values():
            batch_size = tensor.shape[0]
            break

        # Truncated depth standardization
        if standardize_depth and 'depth' in tasks_dict:
            # Flatten depth and remove bottom and top 10% of values
            trunc_depth: Tensor = torch.sort(rearrange(tasks_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
            trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
            tasks_dict['depth'] = (tasks_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) \
                / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with torch.cuda.amp.autocast():
            results = model(
                input_dict, 
                num_encoded_tokens=num_encoded_tokens, 
                alphas=alphas, 
                sample_tasks_uniformly=sample_tasks_uniformly,
                fp32_output_adapters=fp32_output_adapters,
            )
            '''
            {
                'rgb': tensor of shape (b, c, h, w)
                'depth': tensor of shape (b, 1, h, w)
                'norm_rgb': tensor of shape (b, c, h, w)
            }
            '''
            preds: Dict[str, Tensor] = results[0]
            masks: Dict[str, Tensor] = results[1]

            if extra_norm_pix_loss:
                tasks_dict['norm_rgb'] = tasks_dict['rgb']
                masks['norm_rgb'] = masks.get('rgb', None)

            task_losses: Dict[str, Tensor] = {}
            for task in preds:
                target = tasks_dict[task]
                    
                if loss_on_unmasked:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                else:
                    task_losses[task] = tasks_loss_fn[task](
                        preds[task].float(), target, mask=masks.get(task, None))
            loss_value = sum(task_losses.values())
            
            if args.distributed:
                dist.all_reduce(loss_value)
            
            total_loss += loss_value.to(CPU_DEVICE).item() * batch_size
        
    total_loss = total_loss / len(data_loader.dataset)
    if log_writer is not None:
        log_writer.update({f'{dataset_type}_loss': total_loss})


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
