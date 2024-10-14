import os
import numpy as np
import torch 
from torch.utils.data import DataLoader, random_split
import json

from .quality_dataset import QualityDataset
from .quality_model import QualityClassifier

from .utils import set_global_seed

ds = QualityDataset()

n_timesteps = ds.n_timesteps
quality_cache = None
if os.path.exists('../data/quality/quality_all_answers.pt'):
    all_answers = torch.load('../data/quality/quality_all_answers.pt')
    quality_cache = {sample['id']: sample for sample in all_answers['X']}
model = QualityClassifier(n_timesteps, stop_threshold=1.0, cache=quality_cache)

def get_eff_sizes(objs_path, n_data):
    sizes = np.zeros((objs_path.shape[0], objs_path.shape[1]-1))
    sizes[:,0] = n_data
    return sizes


def extract_lambdas(lambdas, config):  
    th = np.zeros(config["n_timesteps"])
    for  i in range(len(config["lambda_names"])):
        if f"th_{i}" in config["lambda_names"] :
            index = config["lambda_names"].index(f"th_{i}")
            print(i, index)
            print(lambdas[index])
            th[i] = lambdas[index]
        else:
            th[i] = 1.0    
    return th

def evaluate(X, y, th, config):
    tmp_res = model.forward(X)
    pi = tmp_res['all_is_correct_estimation'].detach().cpu()
    preds_per_t = tmp_res['all_scores'].detach().cpu().numpy().argmax(axis=-1)
    is_correct_per_t = np.array([int(y[i]) == preds_per_t[i] for i in range(len(y))])
    late_is_correct = is_correct_per_t[:, -1]
    
    should_stop = pi >= th
    should_stop[:, -1] = True  # Always stop at the last time step
    halt_timesteps = should_stop.float().argmax(dim=-1)
    is_correct = is_correct_per_t[torch.arange(is_correct_per_t.shape[0]), halt_timesteps]

    gap_mean = np.maximum(late_is_correct.astype(float) - is_correct.astype(float), 0).mean()
    time_mean = (halt_timesteps+1).float().mean()/n_timesteps 

    return gap_mean, time_mean   

def train(lambdas, config):
    th = torch.from_numpy(lambdas) #extract_lambdas(lambdas, config)
    # th = torch.zeros_like(th0)
    # th[0] = th0[0]
    # for i in range(1,len(config["lambda_names"])):
    #    th[i] = th[i-1] - th0[i]

    set_global_seed(config['seed'])
    val_ds, test_cal_ds = random_split(ds, [config["n_val"], config["n_test"]+config["n_cal"]])
 
    val_X = [x for x, y in val_ds]
    val_y = [y for x, y in val_ds]

    gap_mean, time_mean = evaluate(val_X, val_y, th, config)

    scores = np.array([gap_mean, time_mean])

    return scores

def evaluate_path(random_state, lambdas_path, config):
    scores_cal = np.zeros((lambdas_path.shape[0], len(config["objs_names"])))
    scores_test = np.zeros((lambdas_path.shape[0], len(config["objs_names"])))

    set_global_seed(config['seed'])
    val_ds, test_cal_ds = random_split(ds, [config["n_val"], config["n_test"]+config["n_cal"]])
    set_global_seed(random_state)
    cal_ds, test_ds = random_split(test_cal_ds, [config["n_cal"], config["n_test"]])

    cal_X = [x for x, y in cal_ds]
    cal_y = [y for x, y in cal_ds]

    test_X = [x for x, y in test_ds]
    test_y = [y for x, y in test_ds]

    for i, lambdas in enumerate(lambdas_path):
        th = torch.from_numpy(lambdas)
        # th0 = torch.from_numpy(lambdas) #extract_lambdas(lambdas, config)
        # th = torch.zeros_like(th0)
        # th[0] = th0[0]
        # for j in range(1,len(config["lambda_names"])):
        #     th[j] = th[j-1] - th0[j]
        gap_mean, time_mean = evaluate(cal_X, cal_y, th, config)
        scores_cal[i,:] = np.array([gap_mean, time_mean])
        gap_mean, time_mean = evaluate(test_X, test_y, th, config)
        scores_test[i,:] = np.array([gap_mean, time_mean])

    sizes = get_eff_sizes(scores_cal, config["n_cal"])
    return scores_cal, scores_test, sizes   

def init_configs(lambdas_init, config):    
    N = lambdas_init.shape[0]
    scores =  np.zeros((N,len(config["objs_names"])))    
    for n in range(N):
        print(f'Train round {n}')
        print(r'$\lambda$ =' + f'{lambdas_init[n]}')
        scores[n, :] = train(lambdas_init[n], config)

    return scores


if __name__ == '__main__':
    config_file = 'config.json'
    with open(config_file) as cf_file:
        config= json.loads(cf_file.read())  

    lambdas = np.random.uniform(size=(1,10))
    lambdas = np.ones((1,10)) + 1e-10
    config['seed'] = 0
    scores = train(lambdas, config) 
    print(scores)   

