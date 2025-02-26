import torch
import numpy as np




class SingleOutputEngine:
    
    def __init__(self, model, optimizer, loss_fn):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.train_loader = None
        self.val_loader = None
        
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        
    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def _mini_batch(self, validation=False):
        
        batch_losses = []
        
        step_fn = self._get_train_step_fn()
        loader = self.train_loader
        self.model.train()
        
        if validation:
            step_fn = self._get_eval_step_fn()
            loader = self.val_loader
            self.model.eval()
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            
            loss = step_fn(x, y)
            
            batch_losses.append(loss)
        
        return np.array(batch_losses).mean()
        
    def _get_train_step_fn(self):
        def perform_step_step(x, y):
            logits = self.model(x)
            
            loss = self.loss_fn(logits, y)
            
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        return perform_step_step
    
    def _get_eval_step_fn(self):
        def perform_evel_step(x, y):
            with torch.no_grad():
                logits = self.model(x)
                
                loss = self.loss_fn(logits, y)
                
                return loss.item()
        return perform_evel_step
    
    def set_seed(self, seed=42):
        torch.manual_seed(42)
    
    def train(self, epochs):
        
        self.set_seed()
        
        for epoch in range(epochs):
            
            self.epoch = epoch + 1
            
            self.train_losses.append(self._mini_batch())
            if self.val_loader:
                self.val_losses.append(self._mini_batch(validation=True))
            
    
        