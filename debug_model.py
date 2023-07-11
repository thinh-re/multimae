from domain_conf import get_model
from multimae.multimae import MultiMAE
from pretrain_argparser import get_args
import torch

args = get_args()
model: MultiMAE = get_model(args)

input_dict = dict(
    image=torch.randn(1, 3, args.input_size, args.input_size),
    depth=torch.randn(1, 3, args.input_size, args.input_size),
)
p = model.forward(
    input_dict,
    num_encoded_tokens=args.num_encoded_tokens,
    alphas=args.alphas,
    sample_tasks_uniformly=args.sample_tasks_uniformly,
    fp32_output_adapters=args.fp32_output_adapters,
)
print()