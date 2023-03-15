from copy import deepcopy

import torch

class ModelEMA(object):
    def __init__(self, args, model, ema_decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.ema_decay = ema_decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
    
        print('self.param_keys: {}'.format(self.param_keys))
        print('self.buffer_keys: {}'.format(self.buffer_keys))
        
        for p in self.ema.parameters():
#             print('Inside ModelEMA, p dtype is {}'.format(p.dtype))
            p.requires_grad_(False)
        
        
    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.ema_decay + (1. - self.ema_decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
