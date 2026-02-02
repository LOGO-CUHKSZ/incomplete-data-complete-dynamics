import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, Sequence, Callable

from tqdm import tqdm

from abc import ABC, abstractmethod

from models.learning import *

__all__ = [
    'DM', 'SDEScheule',
]

class SDEGenerative(ABC):
    def __init__(
        self,
        sde_config: dict,
        model_config: dict,
        learning_config: dict,
        dataset
    ):
        super().__init__()
        self.sde_config = sde_config
        self.model_config = model_config
        self.matching = sde_config.matching
        self.sde = SDEScheule(**sde_config)
        
        n_channels, n_frames, image_size, _ = dataset.u_shape

        match learning_config.model:
            case 'ldpv' | 'dsm':
                input_channels = 2 * n_channels
                output_channels = n_channels
            case _:
                raise NotImplementedError()
            
        data_shape = {
            'input_channels': input_channels,
            'output_channels': output_channels,
            'n_frames': n_frames,
            'image_size': image_size
        }

        if model_config.model.lower() == 'karras':
            from models.Unet.unet import Unet3D
            self.network = Unet3D(**(model_config.network | data_shape))
        elif model_config.model.lower() == 'fno':
            from models.FNO import FNO3D
            self.network = FNO3D(**(model_config.network | data_shape))
        else:
            raise NotImplementedError()
        
        print(f"{self.network._get_name()} #Params: {sum(p.numel() for p in self.network.parameters())}")

    def batch_extend_like(self, src: Tensor, out: Tensor) -> Tensor:
        return src.view(src.size(0), *[1] * (out.ndim - 1))

    @abstractmethod
    def get_loss(self, batch: Sequence[Tensor] | Tensor) -> tuple[Dict[str, Tensor], int]:
        pass

    @abstractmethod
    @torch.inference_mode()
    def sampling(self) -> Tensor:
        pass


class DM(nn.Module, SDEGenerative):
    def __init__(
        self,
        sde_config: dict,
        model_config: dict,
        learning_config: dict,
        dataset
    ):
        nn.Module.__init__(self)
        SDEGenerative.__init__(self, sde_config, model_config, learning_config, dataset)
        self.learning = learning_config


    def get_loss(self, batch: Sequence[Tensor]) -> tuple[Dict[str, Tensor], int]:
        loss_dict = {}
        # x0, mask shape  == (batch_size, n_channel, n_time, ..., n_grid)
        # variables.shape == (batch_size, n_var)
        x0, mask = batch

        t, xt, vt, noise = self.sde.dm_forward(x0, mask=mask)

        assert self.learning.model == 'ldpv'

        ctx, ctx_mask, qry_mask = diffusion_fields(xt, mask, **self.learning)
        model_pred: Tensor = self.network(t, ctx, ctx_mask)
        assert model_pred.shape == noise.shape

        loss = ((model_pred - x0)[qry_mask.expand_as(x0)]).square().mean()

        loss_dict['total'] = loss

        return loss_dict, x0.size(0)

    def pred_noise(self, t: Tensor, xt: Tensor, model_input: Sequence[Tensor], **kwargs) -> Tensor:
        batch_t = t.repeat(xt.size(0)) if t.ndim == 0 else t
        alpha = self.batch_extend_like(self.sde.alpha(batch_t), xt)
        sigma = self.batch_extend_like(self.sde.sigma(batch_t), xt)
        model_predict = self.network(batch_t, *model_input, **kwargs)
        return (xt - alpha * model_predict) / sigma

    
    def ode_reverse(
            self,
            s: Tensor, t: Tensor,
            xt: Tensor, pred_noise: Tensor
        ) -> Tensor:
        sigma_t, sigma_s = self.sde.sigma(t), self.sde.sigma(s)
        alpha_t, alpha_s = self.sde.alpha(t), self.sde.alpha(s)

        model_out = sigma_t * (alpha_t * sigma_s / (sigma_t * alpha_s)) * pred_noise
        return alpha_t / alpha_s * xt - model_out
    
    @torch.inference_mode()
    def sampling(
        self,
        x0: Tensor, mask: Tensor,
        steps: int = 100,
        sampling: dict | None = None,
        mask_func: Callable[[Tensor], Tensor] | None = None,
        **kwargs
    ) -> Tensor:
        device = next(self.network.parameters()).device
        ts = torch.linspace(1.0, self.sde.t_eps, steps+1, device=device)
        clip = lambda x: torch.clip(x, -5, 5)   

        if steps == 1:
            start_t = self.sde.t_eps
            t = torch.full((x0.size(0), ), start_t, device=device)
            xt = self.sde.dm_forward(
                x0=x0, mask=mask, t=t
            )[1]

            if sampling is not None and sampling.get('random_context', False):
                assert self.learning.model == 'ldpv'
                pred = []
                for _ in range(10):
                    ctx, ctx_mask, _ = diffusion_fields(
                        xt, mask, **self.learning
                    )
                    pred.append(self.network(t, ctx, ctx_mask, **kwargs))
                xt = torch.stack(pred).mean(0)
            else:
                match self.learning.model:
                    case 'dsm':
                        model_input = [xt, mask]
                    case 'ldpv':
                        ctx, ctx_mask, _ = diffusion_fields(xt, mask, **self.learning)
                        model_input = [ctx, ctx_mask]
                xt = self.network(t, *model_input, **kwargs)

        elif sampling is None:
            assert self.learning.model == 'dsm'
            pbar = tqdm(
                zip(ts[:-1], ts[1:]),
                leave=False, total=len(ts)-1, dynamic_ncols=True
            )
            xt = self.sde.dm_forward(
                x0=x0, mask=mask, t=torch.ones(size=(x0.size(0), ), device=device)
            )[1]
            for s, t in pbar:
                model_input = [mask * xt, mask]
                eps_model = clip(self.pred_noise(s, xt, model_input, **kwargs))
                observed_noise = clip((xt - self.sde.alpha(s) * x0) / self.sde.sigma(s))
                eps_model = mask * observed_noise + ~mask * eps_model
    
                xt = self.ode_reverse(s, t, xt, eps_model)

            model_input = [mask * xt, mask]
            xt = self.network(t.repeat(xt.size(0)), *model_input, **kwargs)

        elif sampling.get('weighted', False):
            assert self.learning.model == 'ldpv'
            
            weight: str = sampling['weighted']
            if weight.startswith('line'):
                weight = lambda x: x
            elif weight.startswith('quad'):
                weight = lambda x: x**2
            else:
                raise NotImplementedError()
            
            pbar = tqdm(
                zip(ts[:-1], ts[1:]),
                leave=False, total=len(ts)-1, dynamic_ncols=True
            )
            xt = self.sde.dm_forward(
                x0=x0, mask=mask, t=torch.ones(size=(x0.size(0), ), device=device)
            )[1]
        
            n_expect = 10
            mu0 = []
            for _ in range(n_expect):
                x0_ = self.sde.dm_forward(
                    x0=x0, mask=mask,
                    t=torch.full((x0.size(0), ), self.sde.t_eps, device=device)
                )[1]
                context, context_mask, _ = diffusion_fields(x0_, mask, **self.learning)
                model_input = [context, context_mask]
                mu0.append(self.network(ts[-1].repeat(xt.size(0)), *model_input, **kwargs)) 
            mu0 = torch.stack(mu0).mean(0)
            assert mu0.shape == x0.shape

            for s, t in pbar:
                model_input = [context, context_mask]
                pred_mut = self.network(s.repeat(xt.size(0)), *model_input, **kwargs)
                model_mu = weight(s) * mu0 + (1 - weight(s)) * pred_mut
                eps_model = clip((xt - self.sde.alpha(s) * model_mu) / self.sde.sigma(s))
                observed_noise = clip((xt - self.sde.alpha(s) * x0) / self.sde.sigma(s))
                eps_model = mask * observed_noise + ~mask * eps_model
                
                xt = self.ode_reverse(s, t, xt, eps_model)

                random_mask = mask_func(xt)
                context, context_mask, _ = diffusion_fields(
                    xt, random_mask, **self.learning
                )

            mu0 = []
            for _ in range(n_expect):
                random_mask = mask_func(xt)
                context, context_mask, _ = diffusion_fields(xt, random_mask, **self.learning)
                model_input = [context, context_mask]
                mu0.append(self.network(t.repeat(xt.size(0)), *model_input, **kwargs)) 
            xt = torch.stack(mu0).mean(0)

        else:
            assert self.learning.model == 'ldpv'
            pbar = tqdm(
                zip(ts[:-1], ts[1:]),
                leave=False, total=len(ts)-1, dynamic_ncols=True
            )
            xt = self.sde.dm_forward(
                x0=x0, mask=mask, t=torch.ones(size=(x0.size(0), ), device=device)
            )[1]
            context, context_mask, _ = diffusion_fields(xt, mask, **self.learning)
            for s, t in pbar:
                model_input = [context, context_mask]
                eps_model = clip(self.pred_noise(s, xt, model_input, **kwargs))
                observed_noise = clip((xt - self.sde.alpha(s) * x0) / self.sde.sigma(s))
                eps_model = mask * observed_noise + ~mask * eps_model

                xt = self.ode_reverse(s, t, xt, eps_model)

                if sampling.get('random_context', False):
                    random_mask = mask_func(xt)
                    context, context_mask, _ = diffusion_fields(
                        xt, random_mask, **self.learning
                    )
                else:
                    context = context_mask * xt

            if sampling.get('random_context', False):
                n_expect = 10
                mu0 = []
                for _ in range(n_expect):
                    random_mask = mask_func(xt)
                    context, context_mask, _ = diffusion_fields(xt, random_mask, **self.learning)
                    model_input = [context, context_mask]
                    mu0.append(self.network(t.repeat(xt.size(0)), *model_input, **kwargs)) 
                xt = torch.stack(mu0).mean(0)
            else:
                model_input = [context_mask * xt, context_mask]
                xt = self.network(t.repeat(xt.size(0)), *model_input, **kwargs)
                
        return ~mask * xt + mask * x0 
    

class SDEScheule():
    def __init__(
            self,
            beta_min: float = 0.1,
            beta_max: float = 20.0,
            t_eps: float = 1e-3,
            **kwargs
        ) -> None:
        assert beta_min < beta_max

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_eps = t_eps


    def log_mean_coeff(self, t: Tensor) -> Tensor:
        return -0.25 * (t ** 2) * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
    
    def d_log_mean_coeff(self, t: Tensor) -> Tensor:
        return -0.5 * t * (self.beta_max - self.beta_min) - 0.5 * self.beta_min
    
    def f(self, t: Tensor) -> Tensor:
        return self.d_log_mean_coeff(t)

    def g_square(self, t: Tensor) -> Tensor:
        sigma_t = self.sigma(t)
        return 2 * sigma_t * (self.d_sigma(t) - self.d_log_mean_coeff(t) * sigma_t)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.exp(self.log_mean_coeff(t))

    def d_alpha(self, t: Tensor) -> Tensor:
        return self.alpha(t) * self.d_log_mean_coeff(t)
    
    def sigma(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - torch.exp(2 * self.log_mean_coeff(t)))
    
    def d_sigma(self, t: Tensor) -> Tensor:
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = torch.sqrt(1 - torch.exp(p_sigma_t))
        return torch.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
    

    ### For DPM solvers
    def lamb(self, t: Tensor) -> Tensor:
        log_mu = self.log_mean_coeff(t)
        return log_mu - 0.5 * (1 - torch.exp(2 * log_mu)).log()
    
    def _quad_solve(self, c: Tensor) -> Tensor:
        a = 0.25 * (self.beta_max - self.beta_min)
        b = 0.5 * self.beta_min
        discriminant = b**2 - 4 * a * c
        return (-b + torch.sqrt(discriminant)) / (2 * a)
    
    def i_log_mean(self, log_mean: Tensor) -> Tensor:
        return self._quad_solve(log_mean)
    
    def i_lamb(self, lamb: Tensor) -> Tensor:
        return self.i_log_mean(lamb - 0.5 * (1 + (2 * lamb).exp()).log())


    def dm_forward(
            self,
            x0: Tensor,
            t: Tensor | None = None,
            noise: Tensor | None = None,
            mask: Tensor | None = None
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        if t is None:
            rand_t = torch.rand(size=(x0.size(0), ), device=x0.device)
            t = self.t_eps + (1 - self.t_eps) * rand_t
        assert t.size(0) == x0.size(0)
        t = t.view(x0.size(0), *[1] * (x0.ndim - 1))

        noise = torch.randn_like(x0) if noise is None else noise

        xt = self.alpha(t) * x0 + self.sigma(t) * noise
        vt = self.d_alpha(t) * x0 + self.d_sigma(t) * noise

        if mask is not None:
            xt = mask * xt
            vt = mask * vt

        return t.view(-1), xt, vt, noise






