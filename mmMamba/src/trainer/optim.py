"""
Optimizer and schedulers
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_optimizer(optim: str, model: nn.Module, **kwargs: any) -> Optimizer:
    """
    Return training optimizer
    """
    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optim == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif optim in ['adamw', 'adamw_torch']:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    elif optim == 'adamw_torch_fused':
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'bias' in name:
                    no_decay.append(param)
                else:
                    decay.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay, 'weight_decay': kwargs.get('weight_decay', 0.0)},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, **kwargs)
    elif optim == 'adafactor':
        from transformers import Adafactor
        kwargs['relative_step'] = False  # for now
        return Adafactor(model.parameters(), **kwargs)
    else:
        raise NotImplementedError(f"{optim} optimizer not implemented sorry.")


def get_scheduler(lr_scheduler_type: str, optimizer: Optimizer, 
                  **kwargs: any) -> LRScheduler:
    """
    Return learning rate scheduler
    """
    if lr_scheduler_type in ['plateau', 'reduce_lr_on_plateau']:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer=optimizer, **kwargs)
    
    elif lr_scheduler_type == 'cosine_warmup':
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer=optimizer, **kwargs)
    
    elif lr_scheduler_type in ['linear_warmup', 'linear']:
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(optimizer=optimizer, **kwargs)
    elif lr_scheduler_type == 'constant_warmup':
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(optimizer=optimizer, **kwargs)
    elif lr_scheduler_type == 'wsd':
        from transformers import get_wsd_schedule
        return get_wsd_schedule(optimizer=optimizer, **kwargs)
    else:
        return None
