# --------------------------------------------------------
# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------
import io
import os
from pathlib import Path
from typing import Dict, Optional, OrderedDict

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer

from pretrain_argparser import PretrainArgparser

from .dist import is_main_process, save_on_master
from .model import get_state_dict
from .multimae_keys import MULTIVIT_PRETRAINED_KEYS
from .native_scaler import NativeScalerWithGradNormCount


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, Tensor],
    prefix="",
    ignore_missing="relative_position_index",
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


def save_model(
    args: PretrainArgparser,
    epoch: int,
    model: DistributedDataParallel,
    model_without_ddp: nn.Module,
    optimizer: Optimizer,
    loss_scaler: NativeScalerWithGradNormCount,
    model_ema: Optional[nn.Module] = None,
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            if model_ema is not None:
                to_save["model_ema"] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
            print("save_on_master", checkpoint_path)
        if is_main_process():
            save_model_for_downstream_tasks(args, epoch, model_without_ddp)
    else:
        client_state = {"epoch": epoch}
        if model_ema is not None:
            client_state["model_ema"] = get_state_dict(model_ema)
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def save_model_for_downstream_tasks(
    args: PretrainArgparser,
    epoch: int,
    model_without_ddp: nn.Module,
):
    output_dir = Path(args.output_dir)
    checkpoint_path = os.path.join(
        output_dir, f"multimae_{args.wandb_run_name}_e{epoch}.pth"
    )
    print("save_model_for_downstream_tasks", checkpoint_path)

    state_dict: OrderedDict[str, Tensor] = model_without_ddp.state_dict()
    selected_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key in MULTIVIT_PRETRAINED_KEYS:
            selected_state_dict[key] = value
    torch.save(OrderedDict(model=selected_state_dict), checkpoint_path)


def auto_load_model(
    args: PretrainArgparser,
    model: DistributedDataParallel,
    model_without_ddp: nn.Module,
    optimizer: Optimizer,
    loss_scaler: NativeScalerWithGradNormCount,
    model_ema: Optional[nn.Module] = None,
):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob

        all_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*.pth"))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split("-")[-1].split(".")[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, "checkpoint-%d.pth" % latest_ckpt)
        print(f"Auto resume checkpoint: {args.resume} epoch {latest_ckpt}")

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu"
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if hasattr(args, "model_ema") and args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")

    else:
        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            print(f"Load pretrained weights from {args.pretrained_weights}")
