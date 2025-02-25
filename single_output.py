import torch





class SingleOutputEngine:
    
    def __init__(self, model, optimizer, loss_fn):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
    def _get_train_step_fn(self):
        def perform_step_fn(x, y):
            logits = self.model(x)
            
            loss = self.loss_fn(logits, y)
            
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        return perform_step_fn