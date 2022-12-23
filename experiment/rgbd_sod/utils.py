import torch
from torch import nn

num_format = "{:,}".format

cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda_available = torch.cuda.is_available()


def count_parameters(model: nn.Module) -> str:
    '''Count the number of parameters of a model'''
    return num_format(sum(p.numel() for p in model.parameters() if p.requires_grad))
