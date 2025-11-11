import os, re, random, copy, itertools, math
import numpy as np
import inspect
import math
from typing import Any, Iterable, Callable, Optional, Union
import torch
from torch import nn
from torchgeo.datasets import RasterDataset
import io
import matplotlib.pyplot as plt
import torch.nn.functional as F

def set_seed(seed: int):
    """Configura a semente para reprodutibilidade."""
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # para usar o pad 'reflect', tem comentar aqui
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    
def single_band_model(conv:nn.Module, weigths:bool = False, input:bool = True):
    if input:
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None))
        if weigths:
            w3 = conv.weight.data
            w1 = w3.mean(dim=1, keepdim=True)
            new_conv.weight.data.copy_(w1)
    else:
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=1,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None))
        if weigths:
            w3 = conv.weight.data
            w1 = w3.mean(dim=0, keepdim=True)
            new_conv.weight.data.copy_(w1)
    return new_conv

