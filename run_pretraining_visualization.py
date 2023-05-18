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

import utils
from demo.app import log_inference
from multimae import MultiMAE
from multimae.criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from pretrain_argparser import PretrainArgparser, get_args
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import COCO_SEMSEG_NUM_CLASSES
from utils.datasets import (
    build_multimae_pretraining_dev_dataset_v2,
    build_multimae_pretraining_train_dataset_v2,
)
from utils.optim_factory import create_optimizer

CPU_DEVICE = torch.device("cpu")

DOMAIN_CONF = {
    "rgb": {
        "channels": 3,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=3),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=3),
        "loss": MaskedMSELoss,
    },
    "depth": {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=1),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=1),
        "loss": MaskedL1Loss,
    },
    "semseg": {
        "num_classes": 133,
        "stride_level": 4,
        "input_adapter": partial(
            SemSegInputAdapter,
            num_classes=COCO_SEMSEG_NUM_CLASSES,
            dim_class_emb=64,
            interpolate_class_emb=False,
        ),
        "output_adapter": partial(
            SpatialOutputAdapter, num_channels=COCO_SEMSEG_NUM_CLASSES
        ),
        "loss": partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}


def get_model(args: PretrainArgparser) -> MultiMAE:
    """Creates and returns model from arguments"""
    print(
        f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}"
    )

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=args.patch_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]["output_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn,
        )
        for domain in args.out_domains
    }

    # Add normalized pixel output adapter if specified
    if args.extra_norm_pix_loss:
        output_adapters["norm_rgb"] = DOMAIN_CONF["rgb"]["output_adapter"](
            stride_level=DOMAIN_CONF["rgb"]["stride_level"],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task="rgb",
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn,
        )

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
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

    args.in_domains = args.in_domains.split("-")
    args.out_domains = args.out_domains.split("-")
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]["loss"](
            patch_size=args.patch_size, stride=DOMAIN_CONF[domain]["stride_level"]
        )
        for domain in args.out_domains
    }

    # Add normalized pixel loss if specified
    if args.extra_norm_pix_loss:
        tasks_loss_fn["norm_rgb"] = DOMAIN_CONF["rgb"]["loss"](
            patch_size=args.patch_size,
            stride=DOMAIN_CONF["rgb"]["stride_level"],
            norm_pix=True,
        )

    # Get dataset
    dataset_train = build_multimae_pretraining_train_dataset_v2(args)
    dataset_dev = build_multimae_pretraining_dev_dataset_v2(args)

    if True:  # args.distributed:
        world_size = utils.get_world_size()
        global_rank = utils.get_rank()
        print("global_rank", global_rank)
        sampler_rank = global_rank
        num_training_steps_per_epoch = (
            len(dataset_train) // args.batch_size // world_size
        )

        sampler_train = DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=sampler_rank,
            shuffle=True,
            drop_last=True,
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
        dataset_train,
        sampler=sampler_train,
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
    print(
        "Number of training examples per epoch = %d"
        % (total_batch_size * num_training_steps_per_epoch)
    )

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = create_optimizer(args, {"model": model_without_ddp})
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
        log_writer.set_step(0)
    else:
        log_writer = None

    if global_rank == 0:
        log_inference(
            model,
            seed,
            dataset_dev,
            log_writer,
            epoch=args.start_epoch,
            num_samples=10,
        )

    if log_writer is not None:
        log_writer.finish()


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
