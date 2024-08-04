import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import os
import utils



def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        # self.betas = betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.T_sampling = schedule_opt['T_sampling']
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, m, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, m, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, m, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, m=m, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_lr, mask_0, continous=False):
        device = self.betas.device
        n = x_lr.size(0)
        noise = torch.randn_like(x_lr)
        skip = self.num_timesteps // self.T_sampling
        seq = range(0, self.num_timesteps, skip)
        x0_preds = []
        xs = [noise]
        mask_preds = []
        b = self.betas
        eta = 0.
        gamma_ori = 0.2
        idx = 0
        # sample_inter = (1 | (self.num_timesteps//10))
        seq_next = [-1] + list(seq[:-1])
        mask = mask_0
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et, mask_1 = self.denoise_fn(torch.cat([x_lr, mask_0, xt], dim=1), t)
            mask = mask_1
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            mask_preds.append(mask.to('cpu'))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_lr) + c2 * et
            xs.append(xt_next.to('cpu'))
        ret_img = xs
        return ret_img[-1], mask_preds[-1]

    @torch.no_grad()
    def p_sample_loop_d(self, x_lr, mask_0, h_hat, continous=False):
        device = self.betas.device
        n = x_lr.size(0)
        noise = torch.randn_like(x_lr)
        skip = self.num_timesteps // self.T_sampling
        seq = range(0, self.num_timesteps, skip)
        x0_preds = []
        xs = [noise]
        mask_preds = []
        b = self.betas
        eta = 0.
        gamma_ori = 0.2
        idx = 0
        # sample_inter = (1 | (self.num_timesteps//10))
        seq_next = [-1] + list(seq[:-1])
        mask = mask_0
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et, mask_1 = self.denoise_fn(torch.cat([x_lr, mask_0, xt], dim=1), t)
            mask = mask_1
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            mask_preds.append(mask.to('cpu'))

            gamma = gamma_ori + idx * 0.15
            xt_next_temp = (x_lr + 1) / (2 * h_hat)
            xt_next_temp = xt_next_temp * 2 - 1
            xtemp = xt_next_temp / (1 + gamma / (h_hat * h_hat)) + (gamma / (h_hat * h_hat + gamma)) * x0_t

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * xtemp + c1 * torch.randn_like(x_lr) + c2 * et

            xs.append(xt_next.to('cpu'))
        ret_img = xs
        # if continous:
        #     return ret_img
        # else:
        return ret_img[-1], mask_preds[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_lr, mask, continous=False):
        return self.p_sample_loop(x_lr, mask, continous)

    @torch.no_grad()
    def super_resolution_d(self, x_lr, mask, h_hat, continous=False):
        return self.p_sample_loop_d(x_lr, mask, h_hat, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [n, c, h, w] = x_start.shape
        # t = np.random.randint(1, self.num_timesteps + 1)
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(x_start.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        b = self.betas
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        e = torch.randn_like(x_start)
        x_noisy = x_start * a.sqrt() + e * (1.0 - a).sqrt()
        x_recon, updated_mask = self.denoise_fn(
            torch.cat([x_in['SR'], x_in['mask'], x_noisy], dim=1), t.float())

        loss = self.loss_func(e, x_recon)
        # avg_channel = torch.mean((x_in['SR']+1)/2, dim=(2,3), keepdim=True)
        # avg_channel_gt = torch.mean((x_in['HR']+1)/2, dim=(2,3), keepdim=True)
        res = (x_in['HR']+1)/2 - (x_in['SR']+1)/2
        res = torch.mean(res, dim=1, keepdim=True)
        # res = res * avg_channel_gt / avg_channel
        res_map = torch.where(res < 0.05, torch.zeros_like(res), torch.ones_like(res))
        # mask = torch.unsqueeze(res_map[0,,:,:,:], dim=1)
        # contour = torch.unsqueeze(contour, dim=1)
        # H_img = utils.tensor2uint(res_map[0, :, :, :])
        # M_img = utils.tensor2uint(x_in['mask'][0, :, :, :])
        # save_img_path = os.path.join('./contour', '{}_predict.png'.format(x_in['Index'][0]))
        # utils.imsave(H_img, save_img_path)
        # save_img_path2 = os.path.join('./contour', '{}_gt.png'.format(x_in['Index'][0]))
        # utils.imsave(M_img, save_img_path2)
        loss_mask = self.loss_func(updated_mask, res_map)
        # loss_mask2 = self.loss_func(updated_mask2, res_map)
        return loss + loss_mask * 0.5

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
