import torch
from .model import FullyConnected

device = 'cuda' if torch.cuda.is_available() else 'cpu'        

class WeightedScalingMethod:

    def __init__(self, objectives, reg_sim=0.0, **kwargs):
        self.objectives = objectives
        self.reg_sim= reg_sim
        self.model = FullyConnected(dim=88).to(device)

    def model_params(self):
        return list(self.model.parameters())


    def step(self, batch, weights):
        self.model.train()
        self.model.zero_grad()
        logits = self.model(batch['data'])
        batch.update(logits)
        loss_total = None
        task_losses = []

        for i,(a, objective) in enumerate(zip(weights, self.objectives)):
            task_loss = objective(batch['data'], batch['logits'], batch['labels'], batch['sensible_attribute'], self.model)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            task_losses.append(task_loss.item() if device=='cuda' else task_loss)
        
        if self.reg_sim == 0.0:
            cossim = 0.0
        else:    
            cossim = torch.nn.functional.cosine_similarity(torch.FloatTensor(task_losses).to(device), weights, dim=0)
            loss_total -= self.reg_sim * cossim
            cossim  = cossim.item()
        
        loss_total.backward()
        return loss_total.item(), task_losses


    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)
