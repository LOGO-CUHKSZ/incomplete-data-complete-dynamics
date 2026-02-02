import torch
import torch.nn as nn

__all__ = ['EMA']

class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param

    @torch.no_grad()
    def apply_ema_weights(self):
        for name, param in self.model.named_parameters():
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore_original_weights(self):
        for name, param in self.model.named_parameters():
            param.copy_(self.shadow[name])
