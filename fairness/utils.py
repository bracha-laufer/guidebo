import torch

def dict_to_cuda(d):
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
