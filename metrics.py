from functools import wraps
from torchmetrics import Metric
from inspect import getfullargspec
import numpy as np
import torch
import pandas as pd


def tensor2img(tensor, out_type=np.float, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()

        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[:, :, :], (2, 0, 1))
            # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()[..., None]
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        else:
            img_np = (img_np * 255.0)
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result

def input_converter(apply_to=None):
    def input_converter_wrapper(old_func):
        @wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(tensor2img(args[i]))
                    else:
                        new_args.append(args[i])

            return old_func(*new_args)
        return new_func

    return input_converter_wrapper

@input_converter(apply_to=('img1', 'img2'))
def correlation_coefficient(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    img1 = pd.Series(img1)
    img2 = pd.Series(img2)

    pearson = img1.corr(img2, method="pearson")
    spearman = img1.corr(img2, method="spearman")
    kendall = img1.corr(img2, method="kendall")

    return pearson, spearman, kendall


class BaseMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.pearson = 0
        self.spearman = 0
        self.kendall = 0
        self.num = 0


    def update(self, pred, label):
        pearson, spearman, kendall = correlation_coefficient(pred, label)
        self.pearson = self.pearson + pearson
        self.spearman = self.spearman + spearman
        self.kendall = self.kendall + kendall

        self.num += 1

    def reset(self):
        self.pearson = 0
        self.spearman = 0
        self.kendall = 0
        self.num = 0

    def compute(self):
        self.pearson = self.pearson/self.num
        self.spearman = self.spearman/self.num
        self.kendall = self.kendall/self.num
        return self.pearson, self.spearman, self.kendall