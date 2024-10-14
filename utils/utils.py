import math
import numpy as np
import random
import torch

def factor_int(n,k):
    vals = [math.ceil(math.pow(n,1/k))]*k
    i = 0
    while np.prod(vals) > float(n):
        print(np.prod(vals))
        vals[i] -= 1
        i += 1
    return vals

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
