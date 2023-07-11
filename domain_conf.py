from multimae.criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from functools import partial
from utils.data_constants import COCO_SEMSEG_NUM_CLASSES
from pretrain_argparser import PretrainArgparser, get_args
from multimae import MultiMAE
from utils.model_builder import create_model


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
}


def get_model(args: PretrainArgparser) -> MultiMAE:
    """Creates and returns model from arguments"""
    print(
        f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}"
    )

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=args.input_patch_size,
            image_size=args.input_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]["output_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=args.input_patch_size,
            # preds_per_patch=args.output_patch_size,
            # embed_dim=args.embed_dim,
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
