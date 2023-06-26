from collections import OrderedDict

from pathlib import Path

import torch
from torch import Tensor
import os

from multimae.criterion import MaskedMSELoss

from domain_conf import DOMAIN_CONF
from pretrain_argparser import PretrainArgparser, get_args
from run_pretraining_multimae_v2 import DataPL, ModelPL


def main(args: PretrainArgparser):
    data_pl = DataPL(args)
    model_pl = ModelPL(args)

    artifacts_path = os.path.join(args.output_dir, "artifacts.ckpt")
    print("Load artifacts from", artifacts_path)
    artifacts = torch.load(artifacts_path)

    rs = OrderedDict()
    for k, v in artifacts["state_dict"].items():
        k: str
        v: Tensor
        if k.startswith("model.output_adapters."):  # skip 'output_adapters.*'
            continue
        if k.startswith("model."):
            rs[k[6:]] = v

    exported_model_path = os.path.join(args.output_dir, f"multimae_{args.version}.pth")
    print("Exported model path:", exported_model_path)
    torch.save(OrderedDict(model=rs), exported_model_path)

    keys_path = os.path.join(args.output_dir, "keys.txt")
    print("Keys path:", keys_path)
    with open(keys_path, "w") as f:
        f.write("\n".join(list(rs.keys())))


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if opts.depth_loss == "mse":
        DOMAIN_CONF["depth"]["loss"] = MaskedMSELoss
    main(opts)
