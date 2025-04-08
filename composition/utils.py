import math
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import get_linear_schedule_with_warmup

def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(), 
        lr=self.config.learning_rate,
        weight_decay=self.config.weight_decay
    )
    
    # Calculate total steps (or use the one from config if available)
    total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    scheduler = {
        'scheduler': get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        ),
        'interval': 'step',
        'frequency': 1
    }
    
    return [optimizer], [scheduler]

def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
    
    # Calculate total steps
    total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
    
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1  # 10% warmup
        ),
        'interval': 'step',
        'frequency': 1
    }
    
    return [optimizer], [scheduler]

def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-7):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
    
    # Calculate total steps
    total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    scheduler = {
        'scheduler': warmup_cosine_scheduler(
            optimizer, 
            warmup_steps=warmup_steps,
            total_steps=total_steps
        ),
        'interval': 'step',
        'frequency': 1
    }
    
    return [optimizer], [scheduler]