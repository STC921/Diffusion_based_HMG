from data_loader.dataset_h36m import DatasetH36M
from data_loader.dataset_humaneva import DatasetHumanEva
from models.temporal_transformer import EMA
from models.ST_Transformer import MotionTransformer
from models.diffusion import Diffusion
from utils.utils import get_motion_shape
from DCT import get_dct_matrix
import numpy as np
import torch
from torch import optim, nn
import time
from copy import deepcopy

from visualization import render_animation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = "humaneva" # h36m or humaneva

def create_model_and_diffusion(t, joint_num, dim, num_frames=240):
    model = MotionTransformer( ######
        input_feats=dim,
        num_frames=num_frames,
        sequence_len=t,
        joint_num=joint_num,
        t_latent_dim=512,
        s_latent_dim=64,
        ff_size=2048,
        num_layers=8,
        num_heads=8,
        dropout=0.2,
        activation="gelu",
    ).to(device)
    diffusion = Diffusion(
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        motion_size=(t, joint_num, dim),
        device=device,
        scheduler='Linear')
    return model, diffusion

def train(dataset, batch_size, model, ema_model):

    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150, 225, 275, 350, 450], gamma=0.9)
    loss_fn = nn.MSELoss()
    gif_dir = "gif"
    save_gif_interval = 10
    num_epoch = 1000
    for epoch in range(num_epoch):
        tic = time.time()
        total_loss = 0
        model.train()
        generator_train = dataset.sampling_generator(num_samples=1000, batch_size=batch_size)
        t_s = time.time()
        print(f"Starting training epoch {epoch}:")

        gen_len = 0

        for traj in generator_train:
            # with torch.no_grad():
            B, T, J, D = traj.shape
            # traj = traj.reshape(B, T, J * D)
            # traj = traj.to(device)
            if isinstance(traj, np.ndarray):  # 检查 x 是否是 numpy 数组
                traj = torch.tensor(traj, dtype=torch.float32, device=device)  # 转换为 Tensor

            t = diffusion.sample_timesteps(n=traj.shape[0]).to(device)  # 采样时间步长
            x_t, noise = diffusion.noise_motion(traj, t)  # 生成加噪数据
            # print(f"noise.requires_grad: {noise.requires_grad}")

            # dct_m = dct_m.to(x_t.dtype)  # 让 idct_m 和 sampled_motion 保持相同的数据类型
            # traj = torch.matmul(dct_m[:20], x_t)

            predicted_noise = model(x_t, t)
            # print(f"predicted_noise.requires_grad: {predicted_noise.requires_grad}")
            loss = loss_fn(predicted_noise, noise)
            # print(f"Loss type: {type(loss)}, requires_grad: {loss.requires_grad}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item() * B
            gen_len += 1

        total_loss /= gen_len

        toc = time.time()
        print(f'epoch {epoch} loss: {total_loss} elapsed {(toc - tic):.2f}s ')

    # torch.save(model.state_dict(), f'2TGCN_{dataset_name}_model.pth')
    torch.save(ema_model.state_dict(), f'2TGCN_{dataset_name}_ema_model.pth')

    return model, ema_model


if __name__ == '__main__':
    #print(get_motion_shape())
    data_shape, num_frames, keep_ratio = get_motion_shape(dataset_name)
    print(data_shape, num_frames, keep_ratio)
    batch_size, t, joint_num, dim = data_shape

    # actions = {'Walking'}
    actions = 'Walking'
    if dataset_name == "humaneva":
        dataset = DatasetHumanEva('test', actions=actions)
    else:
        dataset = DatasetH36M('train', actions=actions)
    model, diffusion = create_model_and_diffusion(t, joint_num, dim, num_frames)

    ema_model = model

    model.load_state_dict(torch.load(f'2TGCN_{dataset_name}_ema_model1.pth'))
    # ema_model.load_state_dict(torch.load(f'2T_{dataset_name}_ema_model.pth'))
    # model, ema_model = train(dataset, batch_size, model, ema_model)

    print("outputing...")
    n_samples = 16
    # keep_ratio=0.20
    # temp = diffusion.sample_ddim_progressive(ema_model, n_samples, keep_ratio=keep_ratio)
    temp = diffusion.sample_ddim_progressive(model, n_samples, keep_ratio=keep_ratio)
    print(temp.shape)
    for i in range(10):

        tempi = temp[i]
        # c, dim = tempi.shape
        # print(tempi.dtype)
        # temp1 = tempi.reshape(125, -1, 3)
        temp2 = tempi.detach().cpu().numpy()
        print(temp2.shape)

        render_animation(dataset.skeleton, temp2, output=f'D:/PycharmProjects/Diffusion_based_HMG/output/{dataset_name}_{i}.gif')
