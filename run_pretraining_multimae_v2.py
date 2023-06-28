from pytorch_lightning.loggers import WandbLogger

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch import Tensor
import os

import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW, Optimizer
from data_augmentation import DataAugmentationV6
from mae_not_pretrained_keys import MAE_NOT_PRETRAINED_KEYS
from multimae.criterion import MaskedMSELoss
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH
from torch.utils.data import Dataset
from PIL import Image

import shutil
import numpy as np
import random
import yaml
from multimae import MultiMAE
from domain_conf import DOMAIN_CONF, get_model
from pretrain_argparser import PretrainArgparser, get_args
from utils.lr import LinearLRRestart


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ):
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: _PATH, map_location: Optional[Any] = None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {
            "model." + key: value for key, value in state_dict.items()
        }
        return checkpoint

    def remove_checkpoint(self, path: _PATH) -> None:
        return super().remove_checkpoint(path)


class ModelPL(pl.LightningModule):
    def __init__(self, args: PretrainArgparser):
        super().__init__()

        self.lr_policy = LinearLRRestart(
            1.0, args.elr / args.blr, args.num_epochs_every_restart
        )
        self.lr_policy.set_epoch(1, args.total_iters_per_epoch)
        self.args = args

        # Model
        self.model: MultiMAE = get_model(args)
        self.load_pretrained_weights()

        # Loss
        self.tasks_loss_fn = {
            domain: DOMAIN_CONF[domain]["loss"](
                patch_size=args.patch_size, stride=DOMAIN_CONF[domain]["stride_level"]
            )
            for domain in args.out_domains
        }

        # Add normalized pixel loss if specified
        if args.extra_norm_pix_loss:
            self.tasks_loss_fn["norm_rgb"] = DOMAIN_CONF["rgb"]["loss"](
                patch_size=args.patch_size,
                stride=DOMAIN_CONF["rgb"]["stride_level"],
                norm_pix=True,
            )

        self.validation_step_outputs = []
        self.num_dev_samples = []

        # This property activates manual optimization.
        self.automatic_optimization = False

        self.save_hyperparameters()

    def load_pretrained_weights(self):
        if self.args.pretrained_weights:
            checkpoint = torch.load(self.args.pretrained_weights)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            print("Load pretrained weights from", self.args.pretrained_weights)

    def forward(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        mask_inputs: bool = True,
        task_masks: Dict[str, torch.Tensor] = None,
        num_encoded_tokens: int = 128,
        alphas: Union[float, List[float]] = 1.0,
        sample_tasks_uniformly: bool = False,
        fp32_output_adapters: List[str] = [],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        return self.model.forward(
            x,
            mask_inputs,
            task_masks,
            num_encoded_tokens,
            alphas,
            sample_tasks_uniformly,
            fp32_output_adapters,
        )

    def forward_loss(
        self, images: Tensor, depths: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # 1. Prepare input dict
        tasks_dict = {"rgb": images, "depth": depths}
        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in self.args.in_domains
        }

        # 2. Forward
        results = self.model(
            input_dict,
            num_encoded_tokens=self.args.num_encoded_tokens,
            alphas=self.args.alphas,
            sample_tasks_uniformly=self.args.sample_tasks_uniformly,
            fp32_output_adapters=self.args.fp32_output_adapters,
        )
        """
        {
            'rgb': tensor of shape (b, c, h, w)
            'depth': tensor of shape (b, 1, h, w)
            'norm_rgb': tensor of shape (b, c, h, w)
        }
        """
        preds: Dict[str, Tensor] = results[0]
        masks: Dict[str, Tensor] = results[1]

        if self.args.extra_norm_pix_loss:
            tasks_dict["norm_rgb"] = tasks_dict["rgb"]
            masks["norm_rgb"] = masks.get("rgb", None)

        # 3. Calculate loss (overall task, each task)
        task_losses: Dict[str, Tensor] = {}
        for task in preds:
            target = tasks_dict[task]

            if self.args.loss_on_unmasked:
                task_losses[task] = self.tasks_loss_fn[task](
                    preds[task].float(), target
                )
            else:
                task_losses[task] = self.tasks_loss_fn[task](
                    preds[task].float(), target, mask=masks.get(task, None)
                )

        loss = sum(task_losses.values())
        return loss, task_losses

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Any], batch_idx):
        opt: Optimizer = self.optimizers()
        lr: LambdaLR = self.lr_schedulers()

        # Different learning rate
        # 1.0 for pretrained weights
        # 100.0 for new weights
        lrs = lr.get_lr()
        for i in range(len(opt.param_groups)):
            opt.param_groups[i]["lr"] = lrs[i] * opt.param_groups[i]["lr_scale"]

        # print(opt.param_groups[0]["lr"], opt.param_groups[1]["lr"])
        images, depths = batch[0]
        loss, task_losses = self.forward_loss(images, depths)

        # Logging
        logging = {f"{task}_loss": l.item() for task, l in task_losses.items()}
        logging["loss"] = loss
        # logging["lr"] = lrs[0]
        self.log_dict(logging, sync_dist=True, prog_bar=True)

        if not torch.isnan(loss):
            opt.zero_grad()
            # automatically applies scaling, etc...
            self.manual_backward(loss)

        if self.args.clip_grad is not None:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.args.clip_grad,
                gradient_clip_algorithm="norm",
            )

        opt.step()
        lr.step()
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Any], batch_idx):
        images, depths = batch
        loss, task_losses = self.forward_loss(images, depths)
        loss = loss.cpu()
        num_samples = images.shape[0]
        self.validation_step_outputs.append(loss * num_samples)
        self.num_dev_samples.append(num_samples)

    def on_validation_epoch_end(self):
        rs = np.sum(self.validation_step_outputs) / np.sum(self.num_dev_samples)
        print("dev_mae", rs)
        self.log_dict({"dev_mae": rs}, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        self.num_dev_samples.clear()

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(self.parameters(), lr=self.args.blr)

        """
        d = {
            "0.01": [{
                "params": Tensor,
                "name": n,
                "lr_scale": ...
            }],
            ...
        }
        """
        grouped_parameters = [
            # Not-pretrained
            {"params": [], "lr": self.args.blr, "lr_scale": self.args.lr_scale},
            # Pretrained
            {"params": [], "lr": self.args.blr, "lr_scale": 1.0},
        ]
        for k, v in self.model.named_parameters():
            if k in MAE_NOT_PRETRAINED_KEYS:
                grouped_parameters[0]["params"].append(v)
            else:
                grouped_parameters[1]["params"].append(v)

        for group in grouped_parameters:
            print(
                f"Group lr_scale={group['lr_scale']}",
                sum([p.numel() for p in group["params"]]),
            )

        optimizer = AdamW(
            grouped_parameters,
            lr=self.args.blr,
            betas=self.args.opt_betas,
            weight_decay=self.args.weight_decay,
        )

        def lr_lambda(current_step: int):
            return self.lr_policy.get_lr(current_step)

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "name": "learning_rate",
            "interval": "step",
        }

        return [optimizer], [scheduler]


def load_json(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


class MDataset(Dataset):
    def __init__(
        self,
        input_size: int,
        data_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.max_samples = max_samples

        raw_data = load_json(os.path.join("datasets_metadata", f"{data_path}.json"))
        self.data: List[Dict[str, Any]] = raw_data[split]["samples"]
        for e in self.data:
            if not e["rgb"].startswith("/"):
                e["rgb"] = os.path.join("datasets", e["rgb"])
            if not e["depth"].startswith("/"):
                e["depth"] = os.path.join("datasets", e["depth"])

        self.data_augmentation = DataAugmentationV6(input_size)

        if self.max_samples is not None:
            self.data = self.data[: self.max_samples]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        data = self.data[index]
        image = Image.open(data["rgb"]).convert("RGB")
        depth = Image.open(data["depth"]).convert("L")

        # Fix mismatch depth and image size (inplace)
        if depth.size != image.size:
            depth = depth.resize(image.size)
            depth.save(data["depth"])

        image, depth = self.data_augmentation.forward(
            image, depth, is_transform=self.split == "train"
        )
        return image, depth

    def __len__(self) -> int:
        return len(self.data)


class DataPL(pl.LightningDataModule):
    def __init__(self, args: PretrainArgparser):
        super().__init__()
        self.args = args

        self.train_dataset = MDataset(
            args.input_size,
            args.data_path,
            split="train",
            # max_samples=100,  # remove this
        )
        self.dev_dataset = MDataset(
            args.input_size,
            args.data_path,
            split="validation",
            # max_samples=100,  # remove this
        )
        self.test_dataset = MDataset(
            args.input_size,
            args.data_path,
            split="test",
            # max_samples=100,  # remove this
        )

        print("TrainDataset", len(self.train_dataset))
        print("DevDataset", len(self.dev_dataset))
        print("TestDataset", len(self.test_dataset))

        args.num_training_samples_per_epoch = len(self.train_dataset)

        self.g = torch.Generator()
        self.g.manual_seed(self.args.seed)

    def train_dataloader(self):
        return [
            DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                # If ``True``, the data loader will copy Tensors
                # into device/CUDA pinned memory before returning them
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=self.g,
                shuffle=True,
            )
        ]

    def val_dataloader(self):
        return [
            DataLoader(
                self.dev_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                # If ``True``, the data loader will copy Tensors
                # into device/CUDA pinned memory before returning them
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=self.g,
                shuffle=False,
            )
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                self.dev_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                # If ``True``, the data loader will copy Tensors
                # into device/CUDA pinned memory before returning them
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=self.g,
                shuffle=False,
            )
        ]

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def main(args: PretrainArgparser):
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as file:
        yaml.dump(args.todict(), file)

    data_pl = DataPL(args)
    model_pl = ModelPL(args)

    loggers: List[Logger] = []
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.wandb_run_name,
        version=args.version,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)

    wb_logger = WandbLogger(
        project=args.wandb_project,
        id=args.wandb_run_name,
        name=args.wandb_run_name,
        config=args.todict(),
        resume="auto",
    )
    loggers.append(wb_logger)

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="dev_mae",
        verbose=True,
        dirpath=args.output_dir,
        filename="artifacts",
        save_top_k=args.save_top_k,
        save_last=True,
        mode="min",
    )

    # custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        # resume_from_checkpoint=config.get("resume_from_checkpoint_path", None),
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        max_epochs=args.epochs,
        devices=args.devices,
        val_check_interval=1.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=16,
        num_sanity_val_steps=0,
        logger=loggers,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_pl, data_pl)


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if opts.depth_loss == "mse":
        DOMAIN_CONF["depth"]["loss"] = MaskedMSELoss
    main(opts)
