from pathlib import Path
import os
from demo.app import inference
from multimae.criterion import MaskedMSELoss

from domain_conf import DOMAIN_CONF
from pretrain_argparser import PretrainArgparser, get_args
from run_pretraining_multimae_v2 import DataPL, ModelPL
import matplotlib.pyplot as plt


def main(args: PretrainArgparser):
    data_pl = DataPL(args)
    model_pl = ModelPL.load_from_checkpoint(
        os.path.join(args.output_dir, "artifacts.ckpt"),
        args=args,
        map_location=None,
    )
    n_row = 10
    n_col = 6
    f, axarr = plt.subplots(n_row, n_col, figsize=(12, 44))

    model_pl.to(f"cuda:{args.devices[0]}")

    for i, (image, depth) in enumerate(data_pl.test_dataset):
        masked_rgb, pred_rgb, rgb, masked_depth, pred_depth, depth = inference(
            model_pl,
            {"rgb": image, "depth": depth},
            num_tokens=15,
            num_rgb=15,
            num_depth=15,
        )
        # rgb_mae = l1loss(Tensor(rgb), Tensor(pred_rgb))
        # depth_mae = l1loss(Tensor(depth), Tensor(pred_depth))
        axarr[i, 0].imshow(rgb)
        axarr[i, 1].imshow(masked_rgb)
        axarr[i, 2].imshow(pred_rgb)

        axarr[i, 3].imshow(depth)
        axarr[i, 4].imshow(masked_depth)
        axarr[i, 5].imshow(pred_depth)
        if i >= 9:
            break

    os.makedirs(f"tmp", exist_ok=True)
    plt.show()
    plt.savefig(f"tmp/eval.png")


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if opts.depth_loss == "mse":
        DOMAIN_CONF["depth"]["loss"] = MaskedMSELoss
    main(opts)
