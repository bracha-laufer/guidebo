import torch
from abc import abstractmethod
from torch.autograd import grad
import numpy as np


class DDPHyperbolicTangentRelaxation():
    def __init__(self, tanh_slope=3.0):
        
        self.threshold = 0.5
        self.isFairnessLoss = True
        self.reg_type = 'tanh'
        self.good_value = 1
        self.tanh_slope = tanh_slope
        
    def _differentiable_round(self, x):
        x = x.float()
        return torch.tanh(self.tanh_slope*(x - self.threshold))/2 + 0.5
    
    def _DP_torch(self, y_true, y_pred, reg):
        if(reg=='tanh'):
            y_pred = self._differentiable_round(y_pred)
            if(self.good_value):
                y_pred = y_pred[y_pred > self.threshold]
            else:
                y_pred = y_pred[y_pred < self.threshold]
        elif(reg=='ccr'):
            if(self.good_value):
                y_pred = y_pred[y_pred > self.threshold]
            else:
                y_pred = y_pred[y_pred < self.threshold]
        elif(y_pred=='linear'):
            y_pred = y_pred
        total = y_true.shape[0] + 1e-7
        return(torch.sum(y_pred)/total)

    def __call__(self, x, y_pred, y_true, a, model=None):
        
        y_pred = torch.squeeze(torch.sigmoid(y_pred))
        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)
        
        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        DP_0 = self._DP_torch(y_true_0, y_pred_0, self.reg_type) 
        DP_1 = self._DP_torch(y_true_1, y_pred_1, self.reg_type) 
        
        return torch.abs((DP_0 - DP_1))

class BinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):
    

    def __call__(self, x, logits, labels, a=None, model=None):
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        if labels.dtype != torch.float:
            labels = labels.float()
        return super().__call__(logits, labels)



class GradientParityAlighnment():  
    def __init__(self):

        self.loss_func = BinaryCrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity()

 
    def __call__(self, x, y_pred, y_true, a=None, model=None):    
        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]

        loss0 = self.loss_func(y_pred_0, y_true_0)
        loss1 = self.loss_func(y_pred_1, y_true_1)
        
        loss_grads0 = grad(loss0, model.parameters(), create_graph=True) 
        loss_grads1 = grad(loss1, model.parameters(), create_graph=True)

        cosine = 0.0
        for param, grad0, grad1 in zip(model.parameters(), loss_grads0, loss_grads1):
            c0 = (param*grad0)**2
            c1 = (param*grad1)**2
            cosine += self.cos(c0.view(1,-1),c1.view(1,-1))

        return -cosine   

class FairMixup():  
    def __init__(self):

        self.loss_func = BinaryCrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity()

 
    def __call__(self, x, y_pred, y_true, a=None, model=None):    
        batch_x_0 = x[a==0]
        batch_x_1 = x[a==1]

        len0 =  batch_x_0.shape[0]
        len1 =  batch_x_1.shape[0]
        

        while len1 > len0:
            perm0 = torch.randperm(len0) 
            batch_x_0 = torch.cat((batch_x_0, batch_x_0[perm0[:len1-len0]]), dim=0)
            len0 =  batch_x_0.shape[0]
            len1 =  batch_x_1.shape[0]
        
        while len0 > len1:
            perm1 = torch.randperm(len1) 
            batch_x_1 = torch.cat((batch_x_1, batch_x_1[perm1[:len0-len1]]), dim=0) 
            len0 =  batch_x_0.shape[0]
            len1 =  batch_x_1.shape[0]

        alpha = 1
        gamma = np.random.beta(alpha, alpha)

        batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
        batch_x_mix = batch_x_mix.requires_grad_(True)

        output = model(batch_x_mix)

        # gradient regularization
        gradx = torch.autograd.grad(output['logits'].sum(), batch_x_mix, create_graph=True)[0]

        batch_x_d = batch_x_1 - batch_x_0
        grad_inn = (gradx * batch_x_d).sum(1)
        E_grad = grad_inn.mean(0)
        loss_reg = torch.abs(E_grad)

        return loss_reg          





