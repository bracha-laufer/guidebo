import numpy as np 
import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pythae.samplers import NormalSampler
import os

from pythae.models import AutoModel
from pythae.models import BetaVAE, BetaVAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST

import torch.nn.functional as F
from pythae.data.preprocessors import BaseDataset, DataProcessor

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


class VAE:
    '''
    PythAE VAE model
    '''
    def __init__(self, latent_dim, beta, config) -> None:
        
        self.beta = beta
        self.reconstruction_loss = config["reconstruction_loss"]
        self.base_folder = os.path.join(config["checkpoints_dir"], f'seed_{config["seed"]}')
        self.latent_dim = latent_dim
    
        self.model_config = BetaVAEConfig(
                input_dim=(1, 28, 28),
                latent_dim=latent_dim, #16
                beta = beta,
                reconstruction_loss = self.reconstruction_loss
            )

        self.train_config = BaseTrainerConfig(
            output_dir=self.base_folder,
            learning_rate=1e-4,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_epochs=10, # Change this to train the model a bit more
            optimizer_cls="AdamW",
            optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)}
        )    

        self._init_model()
        self._init_pipline()

    def _init_pipline(self):
        self.pipeline = TrainingPipeline(
                training_config=self.train_config,
                model=self.model
            )
    
    def _init_model(self):
        self.model = BetaVAE(
                model_config=self.model_config,
                encoder=Encoder_ResNet_VAE_MNIST(self.model_config), 
                decoder=Decoder_ResNet_AE_MNIST(self.model_config) 
            )  

    def train(self, train_dataset, eval_dataset): 
        
        self.pipeline(
            train_data=train_dataset,
            eval_data=eval_dataset
        )

        return self.evaluate_val(eval_dataset = eval_dataset)

    def evaluate_val(self, eval_dataset):
        
        eval_loss_all = evaluate(self.model, eval_dataset)
        eval_loss = {}
            
        for key, value in eval_loss_all.items():
            eval_loss[key] = value.mean(axis=0)
        
        folders_training = sorted(os.listdir(self.base_folder)) 
        
        for i, folder_name in enumerate(folders_training):
            model_path = os.path.join(self.base_folder, folder_name, 'final_model')
            if os.path.exists(os.path.join(model_path)):
                with open(os.path.join(model_path, "model_config.json")) as f:
                    model_config = json.load(f)

#                 model_beta = model_config["beta"]
#                 if model_beta == self.beta:
#                     print(self.beta, model_path)
#                     model_name = model_path
#                     break

#         json_eval_path = os.path.join(model_name, 'eval_loss.json')
#         with open(json_eval_path, 'w') as f:
#             json.dump(eval_loss, f)        

#         os.rename(os.path.join(self.base_folder, folder_name), 
#                   os.path.join(self.base_folder, f'log_beta_{np.round(np.log10(model_beta),1)}')) 


                if  model_config["beta"] == self.beta and model_config["latent_dim"] == self.latent_dim :
                    print(self.beta, self.latent_dim, model_path)
                    model_name = model_path
                    break
        
        json_eval_path = os.path.join(model_name, 'eval_loss.json')
        with open(json_eval_path, 'w') as f:
            json.dump(eval_loss, f)        
        
        os.rename(os.path.join(self.base_folder, folder_name), 
                  os.path.join(self.base_folder, f'log_beta_{np.round(np.log10(model_config["beta"]),1)}_latent_dim_{model_config["latent_dim"]}')) 
        
        return eval_loss 

def get_eff_sizes(objs_path, n_data):
    sizes = np.zeros((objs_path.shape[0], objs_path.shape[1]-1))
    sizes[:,0] = n_data
    return sizes

def get_modelpath(lambdas, config):
    lambdas_r = []
    for i, (lam, lam_name) in enumerate(zip(lambdas,config["lambda_names"])):
        if lam_name == "log_beta":
            lambdas_r.append(np.round(lam,1))
            if lambdas_r[i] == -0.0:
                lambdas_r[i] = 0.0 
        elif lam_name == "latent_dim":
            lambdas_r.append(np.round(lam,0).astype(int))
        
    #lambdas = np.array([lam if lam!=-0.0 else 0.0 for lam in lambdas])
    # print(lambdas)
    logdir = config["checkpoints_dir"]
    filename = "_".join([f'{name}_{lambdas_r[n]}' for n, name in enumerate(config['lambda_names'])])
    modelpath = os.path.join(logdir,f'seed_{config["seed"]}',filename)
    return modelpath

def extract_lambdas(lambdas, config):      
    if "log_beta" in config["lambda_names"] :
        index = config["lambda_names"].index("log_beta")
        log_beta = lambdas[index]
    else:
        log_beta = -2.0
        
        
    if "latent_dim" in config["lambda_names"] :
        index = config["lambda_names"].index("latent_dim")
        latent_dim= lambdas[index]
    else:
        latent_dim = 16       

    return log_beta, latent_dim

def train(lambdas, config):
    modelpath = get_modelpath(lambdas, config)
    print(lambdas, modelpath)
    if modelpath == f'checkpoints_vae/seed_{config["seed"]}/log_beta_-0.0':
        modelpath =  f'checkpoints_vae/seed_{config["seed"]}/log_beta_0.0'
        print(modelpath)   
    if os.path.exists(modelpath):
        print('Model exists')
        json_eval_path = os.path.join(modelpath,'final_model/eval_loss.json')
        with open(json_eval_path, 'r') as f:
            eval_loss = json.load(f)      
    else:   
        print('Training model...')
        mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
        mnist_dataset = mnist_trainset.data.reshape(-1, 1, 28, 28) / 255
        
        train_dataset, eval_dataset = train_test_split(mnist_dataset, test_size=10000, random_state=config['seed'])

        log_beta, latent_dim = extract_lambdas(lambdas, config)
        vae = VAE(latent_dim = np.round(latent_dim,0).astype(int), beta=np.power(10,np.round(log_beta,1)), config=config)
        eval_loss = vae.train(train_dataset, eval_dataset)
    
    scores = np.zeros(len(config["objs_names"]))
    for s, score_name in enumerate(config["objs_names"]):
        scores[s] = eval_loss[score_name]
    return scores  

def init_configs(lambdas_init, config):    
    N = lambdas_init.shape[0]
    scores =  np.zeros((N,len(config["objs_names"])))    
    for n in range(N):
        print(f'Train round {n}')
        print(r'$\lambda$ =' + f'{lambdas_init[n]}')
        scores[n, :] = train(lambdas_init[n], config)
    return scores

def evaluate_path(random_state, lambdas_path, config):    
    scores_cal, scores_test =  np.zeros((lambdas_path.shape[0],len(config["objs_names"]))), np.zeros((lambdas_path.shape[0],len(config["objs_names"])))    
    for i, lambdas in enumerate(lambdas_path):
        scores_cal[i], scores_test[i] = evaluate_from_config(lambdas, random_state, config)
    
    sizes = get_eff_sizes(scores_cal, config["n_cal"])
    return scores_cal, scores_test, sizes

def set_inputs_to_device(inputs):
        inputs_on_device = inputs

        if device == "cuda":
            cuda_inputs = dict.fromkeys(inputs)

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    cuda_inputs[key] = inputs[key].cuda()

                else:
                    cuda_inputs[key] = inputs[key]
            inputs_on_device = cuda_inputs

        return inputs_on_device

def evaluate(model, dataset):
    
    data_processor = DataProcessor()
    dataset = data_processor.process_data(dataset)
    dataset = data_processor.to_dataset(dataset)

    dataloader = DataLoader(
                dataset=dataset,
                batch_size=64,
            )
    model.eval()

    recon_loss = 0
    kld = 0
    mse_loss = 0
    
    model.eval().to(device)
    
    mse_loss_all, bce_loss_all, KLD_all = np.zeros(len(dataset)), np.zeros(len(dataset)), np.zeros(len(dataset))

    i = 0
    for x in tqdm(dataloader):

        with torch.no_grad():
            x = set_inputs_to_device(x)
            
            x = x["data"]
            
            b_size = x.shape[0]

            
            encoder_output = model.encoder(x)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance

            std = torch.exp(0.5 * log_var)
            z, eps = model._sample_gauss(mu, std)
            recon_x = model.decoder(z)["reconstruction"]
            
            mse_loss_all[i:i+b_size] = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1).detach().cpu().numpy()

            bce_loss_all[i:i+b_size] = F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1).detach().cpu().numpy()

            KLD_all[i:i+b_size] = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).detach().cpu().numpy()
            
            i += b_size
     

    eval_loss = {'BCE': bce_loss_all,
                'KLD': KLD_all,
                'MSE': mse_loss_all/728}

    return eval_loss

def evaluate_from_config(lambdas, random_state, config):
    model_path = get_modelpath(lambdas, config)
    model = AutoModel.load_from_folder(os.path.join(model_path,'final_model')) 
    
    mnist_testset = datasets.MNIST(root='../../data', train=False, download=True, transform=None)
    test_dataset = mnist_testset.data.reshape(-1, 1, 28, 28) / 255.  
    
    filename = 'cal_test_scores.npz'
    filepath = os.path.join(model_path,filename)
    
    if os.path.exists(filepath):
        eval_loss = np.load(filepath)    
    else:
        eval_loss = evaluate(model, test_dataset)
        np.savez(filepath, **eval_loss)
        
        
    splits = ['cal','test']
    scores = {split: np.zeros(len(config["objs_names"])) for split in splits}    
    
    for s, score_name in enumerate(config["objs_names"]):
        score_cal, score_test = train_test_split(eval_loss[score_name], test_size=0.5, random_state=random_state)
        scores['cal'][s] = score_cal.mean(axis=0)
        scores['test'][s] = score_test.mean(axis=0)
            
    return scores['cal'], scores['test']     

# def evaluate(model, dataset):

#     data_processor = DataProcessor()
#     dataset = data_processor.process_data(dataset)
#     dataset = data_processor.to_dataset(dataset)

#     dataloader = DataLoader(
#                 dataset=dataset,
#                 batch_size=64,
#             )
#     model.eval()

#     recon_loss = 0
#     kld = 0
#     mse_loss = 0

#     model.eval().to(device)

#     for x in tqdm(dataloader):

#         with torch.no_grad():
#             x = set_inputs_to_device(x)

#             output = model(x)
#             recon_x = output.recon_x

#             mse = F.mse_loss(
#                     recon_x.reshape(x["data"].shape[0], -1),
#                     x["data"].reshape(x["data"].shape[0], -1),
#                     reduction="none",
#                 ).sum(dim=-1).mean(dim=0)

#         mse_loss += mse.detach().cpu().item()
#         recon_loss += output.recon_loss.detach().cpu().item()
#         kld += output.reg_loss.detach().cpu().item()

#     recon_loss /= len(dataloader)
#     kld /= len(dataloader)
#     mse_loss /= len(dataloader)

#     eval_loss = {'BCE': recon_loss,
#                 'KLD': kld,
#                 'MSE': mse_loss/728}

#     return eval_loss

# def evaluate_from_config(lambdas, random_state, config):
#     model_path = get_modelpath(lambdas, config)
#     model = AutoModel.load_from_folder(os.path.join(model_path,'final_model')) 

#     mnist_testset = datasets.MNIST(root='../../data', train=False, download=True, transform=None)
#     test_dataset = mnist_testset.data.reshape(-1, 1, 28, 28) / 255.  

#     datasets_dic = {}
#     datasets_dic['cal'], datasets_dic['test'] = train_test_split(test_dataset, test_size=0.5, random_state=random_state)

#     splits = ['cal','test']
#     scores = {split: np.zeros(len(config["objs_names"])) for split in splits}
#     for split in splits:
#         eval_loss = evaluate(model, datasets_dic[split])
#         for s, score_name in enumerate(config["objs_names"]):
#             scores[split][s] = eval_loss[score_name]

#     return scores['cal'], scores['test']                




