import os
import numpy as np
import torch
from torch.utils import data
import json


from .dataset import ADULT
from .losses import DDPHyperbolicTangentRelaxation, BinaryCrossEntropyLoss, FairMixup
from .objectives import DPDiff
from .method import WeightedScalingMethod
from .utils import dict_to_cuda


device = 'cuda' if torch.cuda.is_available() else 'cpu'  

def get_eff_sizes(objs_path, n_data):
    sizes = np.zeros((objs_path.shape[0], objs_path.shape[1]-1))
    sizes[:,0] = n_data
    return sizes

def get_modelpath(lambdas, config):
    lambdas = np.round(lambdas,2)
    logdir = config["checkpoints_dir"]
    filename = "_".join([f'{name}_{lambdas[n]}' for n, name in enumerate(config['lambda_names'])])
    modelpath = os.path.join(logdir,f'seed_{config["seed"]}',filename)
    return modelpath

def evaluate(method, data_loader):  
    score_values = np.array([])
    ii = 0
    len_dataset = len(data_loader.dataset)
    preds_all = np.zeros(len_dataset)
    for batch in data_loader:
        if torch.cuda.is_available():
            batch = dict_to_cuda(batch)
        l = method.eval_step(batch['data'])
        logits = l["logits"].cpu().numpy()
        preds = np.squeeze((logits > 0.0).astype(np.int32), axis=1)
        preds_all[ii:(ii+preds.shape[0])] = preds
        ii += preds.shape[0]
        
    labels = data_loader.dataset.y.numpy()
    att = data_loader.dataset.s.numpy()
    err = 1-np.sum(preds_all == labels)/len_dataset

    metric = DPDiff()
    fair = metric.evaluate(np.hstack((labels.reshape(-1,1),att.reshape(-1,1))), preds_all.reshape(-1,1))
    score_values =  np.array([err, fair])
    
    return score_values 

def extract_lambdas(lambdas, config):  
    if "weight_bce" in config["lambda_names"] :
        index = config["lambda_names"].index("weight_bce")
        weight_bce = lambdas[index]
        weights = torch.Tensor(np.array([weight_bce,1.0 - weight_bce])).to(device).float()
    else:
        weights = torch.Tensor([0.5,0.5])
    
    if "reg_sim" in config["lambda_names"] :
        index = config["lambda_names"].index("reg_sim")
        reg_sim = lambdas[index]
    else:
        reg_sim = 0.0

    if "tanh_slope" in config["lambda_names"] :
        index = config["lambda_names"].index("tanh_slope")
        tanh_slope = lambdas[index]
    else:
        tanh_slope = 3.0  

    if "reg_mixup" and "reg_ddp" in config["lambda_names"]:
        reg_mixup = lambdas[config["lambda_names"].index("reg_mixup")]
        reg_ddp = lambdas[config["lambda_names"].index("reg_ddp")]
        weights  = torch.Tensor(np.array([1.0, reg_ddp, reg_mixup])).to(device).float()    

    return weights, reg_sim, tanh_slope      

def train(lambdas, config):
    modelpath = get_modelpath(lambdas, config)

    weights, reg_sim, tanh_slope = extract_lambdas(lambdas, config)

    if os.path.exists(os.path.join(modelpath,'checkpoint')):
        print('Model exists')
        checkpoint = torch.load(os.path.join(modelpath,'checkpoint'),map_location='cuda:0')
        val_scores = checkpoint['val_scores']
    else:
        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
        
        train_set = ADULT(split='train',seed = config["seed"]) 
        val_set = ADULT(split='val', seed = config["seed"])

        train_loader = data.DataLoader(train_set, config['batch_size'], shuffle=True,num_workers=config['num_workers'])
        val_loader = data.DataLoader(val_set, config['batch_size'], shuffle=False,num_workers=config['num_workers'])

        objectives = [BinaryCrossEntropyLoss(), DDPHyperbolicTangentRelaxation(tanh_slope=tanh_slope)]


        if "reg_mixup" in config["lambda_names"]:
            objectives.append(FairMixup())   
        
        method = WeightedScalingMethod(objectives=objectives, reg_sim=reg_sim)

        # main      
        optimizer = torch.optim.Adam(method.model_params(), config['lr'])
        if config['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config['scheduler_milestones'], gamma=config['scheduler_gamma'])
        
        best_scores = np.ones(2)
        for e in range(config['epochs']):
            #print(f"Epoch {e}")

            cum_bce_loss = 0.0
            cum_fair_loss = 0.0

            for b, batch in enumerate(train_loader):
                if torch.cuda.is_available():
                    batch = dict_to_cuda(batch)
                optimizer.zero_grad()
                loss, losses = method.step(batch, weights)
                optimizer.step()

                cum_bce_loss += losses[0]
                cum_fair_loss += losses[1]

            cum_bce_loss /= len(train_loader) 
            cum_fair_loss /= len(train_loader) 

            if config['use_scheduler']:
                scheduler.step()

        val_scores = evaluate(method, val_loader)
        print(val_scores)
        checkpoint = {
            'state_dict': method.model.state_dict(),
            'weights': weights.detach().cpu().numpy(),
            'val_scores': val_scores
        }
        torch.save(checkpoint, os.path.join(modelpath,'checkpoint'))
    return val_scores

def evaluate_path(random_state, lambdas_path, config):
    cal_set = ADULT(split='cal', random_state=random_state, seed = config["seed"])
    test_set = ADULT(split='test', random_state=random_state, seed = config["seed"])

    cal_loader = data.DataLoader(cal_set, config['batch_size'],num_workers=config['num_workers'], shuffle=False)
    test_loader = data.DataLoader(test_set, config['batch_size'],num_workers=config['num_workers'], shuffle=False)

    
    cal_scores = np.zeros((lambdas_path.shape[0], 2))
    test_scores = np.zeros((lambdas_path.shape[0], 2))
    
    for i, lambdas in enumerate(lambdas_path):
        weights, reg_sim, tanh_slope = extract_lambdas(lambdas, config)
        objectives = [BinaryCrossEntropyLoss(), DDPHyperbolicTangentRelaxation(tanh_slope=tanh_slope)]
        
        if "reg_mixup" in config["lambda_names"]:
            objectives.append(FairMixup())  

        method = WeightedScalingMethod(objectives=objectives, reg_sim=reg_sim)
        modelpath = get_modelpath(lambdas, config)
        checkpoint = torch.load(os.path.join(modelpath,'checkpoint'),map_location='cuda:0')
        method.model.load_state_dict(checkpoint['state_dict'])

        cal_scores[i,:] = evaluate(method, cal_loader)  
        test_scores[i,:] = evaluate(method, test_loader)
    
    sizes = get_eff_sizes(cal_scores, config["n_cal"])

    return cal_scores, test_scores, sizes

def init_configs(lambdas_init, config):
    logdir = config["checkpoints_dir"]
    if not os.path.exists(logdir):
        os.makedirs(logdir)
           
    N = lambdas_init.shape[0]
    scores = np.zeros((N,2))
    for n in range(N):
        print(f'Train round {n}')
        print(r'$\lambda$ =' + f'{lambdas_init[n]}')
        scores[n,:] = train(lambdas_init[n], config)

    return scores     
