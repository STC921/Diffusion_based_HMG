import torch
import numpy as np
import math
import torch.fft
# from utils import *


def sqrt_beta_schedule(timesteps, s=0.0001):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def dct_filter(x, keep_ratio=0.9):
    B, T, D = x.shape
    x_dct = torch.fft.fft(x, dim=1)  # 对时间维度进行 DCT 变换

    # 创建一个掩码，仅保留低频部分
    keep_freqs = int(T * keep_ratio)  # 计算要保留的频率数量
    mask = torch.zeros_like(x_dct)
    mask[:, :keep_freqs, :] = 1  # 仅保留低频部分

    x_dct_filtered = x_dct * mask  # 过滤掉高频成分
    x_smooth = torch.fft.ifft(x_dct_filtered, dim=1).real  # 逆变换回时域，并取实部

    return x_smooth


class Diffusion:
    def __init__(self, noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 motion_size=(35, 66),
                 device="cuda",
                 scheduler='Cosine'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.motion_size = motion_size
        self.device = device
        self.scheduler = scheduler  # 'Cosine', 'Sqrt', 'Linear'
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        if self.scheduler == 'Linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.scheduler == 'Cosine':
            return cosine_beta_schedule(self.noise_steps)
        elif self.scheduler == 'Sqrt':
            return sqrt_beta_schedule(self.noise_steps)
        else:
            raise NotImplementedError(f"unknown scheduler: {self.scheduler}")

    def noise_motion(self, x, t):
        """
        前向过程（加噪）： q(x_t | x_0)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        noise = torch.randn_like(x)  # 生成标准正态分布的噪声
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        """
        采样时间步长
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_ddim_progressive(self, model, n_samples, noise=None, keep_ratio=0.25):
        """
        逆向过程（去噪）：从纯噪声生成数据
        """
        # model.eval()
        # if noise is not None:
        #     x = noise
        # else:
        #     # x = torch.randn((n_samples, self.motion_size[0], self.motion_size[1])).to(self.device)
        #     x = torch.randn((n_samples, self.motion_size[0], self.motion_size[1])).to(self.device)
        #
        # with torch.no_grad():
        #     for i in reversed(range(1, self.noise_steps)):
        #         t = torch.full((n_samples,), i, dtype=torch.long).to(self.device)
        #
        #         alpha_hat = self.alpha_hat[t][:, None, None]
        #         alpha_hat_prev = self.alpha_hat[t - 1][:, None, None]
        #
        #         predicted_noise = model(x, t)
        #
        #         predicted_x0 = (x - torch.sqrt(1. - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
        #         pred_dir_xt = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
        #         x = torch.sqrt(alpha_hat_prev) * predicted_x0 + pred_dir_xt  # 采样去噪
        #
        # return x

        model.eval()
        if noise is not None:
            x = noise
        else:
            x = torch.randn((n_samples, self.motion_size[0], self.motion_size[1])).to(self.device)

        prev_predicted_noise = None  # 记录上一时刻的噪声
        ema_beta = 0.9  # EMA 平滑参数

        with torch.no_grad():
            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((n_samples,), i, dtype=torch.long).to(self.device)

                alpha_hat = self.alpha_hat[t][:, None, None]
                alpha_hat_prev = self.alpha_hat[t - 1][:, None, None]

                predicted_noise = model(x, t)

                # 加入 EMA 平滑 predicted_noise，降低噪声抖动
                if prev_predicted_noise is not None:
                    predicted_noise = ema_beta * prev_predicted_noise + (1 - ema_beta) * predicted_noise
                prev_predicted_noise = predicted_noise  # 记录当前噪声

                predicted_x0 = (x - torch.sqrt(1. - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_dir_xt = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                x = torch.sqrt(alpha_hat_prev) * predicted_x0 + pred_dir_xt  # 采样去噪

        x = dct_filter(x, keep_ratio=keep_ratio)

        return x

        # yield x

    def sample_ddim(self, model, n_samples, noise=None):
        final = None
        for sample in self.sample_ddim_progressive(model, n_samples, noise):
            final = sample

        return final
