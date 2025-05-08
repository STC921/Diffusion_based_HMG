import torch
import numpy as np
import math
import torch.fft
from models.ST_Transformer import MotionTransformer
from DCT import get_dct_matrix
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

# def dct_filter(x, keep_ratio=0.9):
#     B, T, D = x.shape
#     x_dct = torch.fft.fft(x, dim=1)  # 对时间维度进行 DCT 变换
#
#     # 创建一个掩码，仅保留低频部分
#     keep_freqs = int(T * keep_ratio)  # 计算要保留的频率数量
#     mask = torch.zeros_like(x_dct)
#     mask[:, :keep_freqs, :] = 1  # 仅保留低频部分
#
#     x_dct_filtered = x_dct * mask  # 过滤掉高频成分
#     x_smooth = torch.fft.ifft(x_dct_filtered, dim=1).real  # 逆变换回时域，并取实部
#
#     return x_smooth

# def dct_filter(x, keep_ratio=0.9):
#     B, T, D = x.shape
#     dct_m, idct_m = get_dct_matrix(T, is_torch=True)  # 获取 DCT 变换矩阵
#
#     dct_m = dct_m.to(x.device).to(x.dtype)
#     idct_m = idct_m.to(x.device).to(x.dtype)
#
#     x_dct = torch.matmul(dct_m, x)  # 计算 DCT 变换 (B, T, D)
#
#     # 创建一个掩码，仅保留低频部分
#     keep_freqs = int(T * keep_ratio)  # 计算要保留的频率数量
#     mask = torch.zeros_like(x_dct)
#     mask[:, :keep_freqs, :] = 1  # 仅保留低频部分
#
#     x_dct_filtered = x_dct * mask  # 过滤掉高频成分
#
#     x_smooth = torch.matmul(idct_m, x_dct_filtered)  # 逆 DCT 变换 (B, T, D)
#
#     return x_smooth

def dct_filter(x, keep_ratio=0.9):
    """
    x: (B, T, J, D)
    只对时间维度 T 做 DCT 变换和滤波，保持 J 和 D 不变
    """
    B, T, J, D = x.shape
    dct_m, idct_m = get_dct_matrix(T, is_torch=True)  # 获取 DCT 变换矩阵
    dct_m = dct_m.to(x.device).to(x.dtype)  # (T, T)
    idct_m = idct_m.to(x.device).to(x.dtype)

    # reshape 为 (B * J * D, T)，方便做矩阵乘法
    x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, T)  # (B*J*D, T)

    x_dct = torch.matmul(dct_m, x_reshaped.T).T  # 先做转置为 (T, B*J*D)，再转回 (B*J*D, T)

    # 创建掩码，仅保留低频部分
    keep_freqs = int(T * keep_ratio)
    mask = torch.zeros_like(x_dct)
    mask[:, :keep_freqs] = 1
    x_dct_filtered = x_dct * mask

    # 逆 DCT 还原
    x_smooth = torch.matmul(idct_m, x_dct_filtered.T).T  # (B*J*D, T)

    # reshape 回 (B, T, J, D)
    x_smooth = x_smooth.view(B, J, D, T).permute(0, 3, 1, 2).contiguous()  # (B, T, J, D)

    return x_smooth


class Diffusion:
    def __init__(self, noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 motion_size=(50, 21, 3),
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
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
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
            x = torch.randn((n_samples, self.motion_size[0], self.motion_size[1], self.motion_size[2])).to(self.device)

        prev_predicted_noise = None  # 记录上一时刻的噪声
        ema_beta = 0.9  # EMA 平滑参数

        with torch.no_grad():
            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((n_samples,), i, dtype=torch.long).to(self.device)

                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prev = self.alpha_hat[t - 1][:, None, None, None]

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


def main():
    # 参数定义
    batch_size = 4
    seq_len = 50
    joint_num = 21
    feat_dim = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化 Diffusion 模型
    diffusion = Diffusion(
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        motion_size=(seq_len, joint_num, feat_dim),
        device=device
    )

    # 初始化动作预测模型（你的模型应支持输入 (B, T, J, 3) 和 timestep）
    model = MotionTransformer(
        input_feats=feat_dim,
        sequence_len=seq_len,
        joint_num=joint_num
    ).to(device)

    # 测试输入数据
    x0 = torch.randn(batch_size, seq_len, joint_num, feat_dim).to(device)
    t = diffusion.sample_timesteps(batch_size).to(device)

    # 正向加噪测试
    x_t, noise = diffusion.noise_motion(x0, t)
    print("Forward (q): x_t shape =", x_t.shape)

    # 逆向采样测试（从随机噪声生成）
    with torch.no_grad():
        x_gen = diffusion.sample_ddim_progressive(model, n_samples=batch_size)
    print("Reverse (p): generated x shape =", x_gen.shape)


if __name__ == '__main__':
    main()