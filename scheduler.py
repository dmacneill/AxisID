"""
Documentation at https://github.com/dmacneill/AxisID. 
"""

class Scheduler():
    
    """Updates and tracks the optimizer parameters during training.
    
    Attributes:
        optimizer: instance of torch.optim.AdamW or torch.optim.SGD
        step_size: number of epochs at each learning rate value
        gamma: decay factor
        lr: current learning rate
        epoch: track the current epoch
    """
    
    def __init__(self, optimizer, params):
        
        step_size, gamma = params
        
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.lr = optimizer.param_groups[0]['lr']
        self.epoch = 0
        
    def step(self):
        
        """Increment the optimizer parameters. Called at the end of each epoch during training.
        """
        if (self.epoch+1)%self.step_size==0:
            self.lr *= self.gamma
        
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.epoch+=1