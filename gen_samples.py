import sys, os
import torch
from tqdm import tqdm
from torch import nn
import yaml
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import matplotlib.pyplot as plt
from sampler import MNISTSampler
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from torchvision.utils import save_image
from sampler import MNISTSampler
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
parser.add_argument('--device',type = str ,default='cuda:7')
args = parser.parse_args()

num_samples = 1000
def process(im):
    return torch.flatten(im, start_dim=0, end_dim=1).detach().cpu().numpy()

device = args.device
torch.set_default_device(device)

print("LOADING MODEL")

lr_model=MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=64,
                dim_mults=[2,4]).to(device)
lr_model_ema = ExponentialMovingAverage(lr_model, device=device, decay=1-0.32)
lr_ckpt = torch.load('models/mnist.pt')
lr_model_ema.load_state_dict(lr_ckpt["model_ema"])
lr_model_ema.eval()

print("GENERATING SAMPLES")
@torch.no_grad()
def sample(model_ema, n_samples, device, seed=2025):
    torch.manual_seed(seed)
    return model_ema.module.sampling(n_samples,clipped_reverse_diffusion=True,device=device)

for iter in range(50):
    samples = sample(lr_model_ema, num_samples, device)

    print("SAVING FILES")

    for i in range(len(samples)):
        if i % 100 == 0:
            print(f"{i//100}/10")
        image_filename = os.path.join('datasets/gen_mnist_images', f'image_{i+1000*(iter)}.png')
        save_image(samples[i], image_filename)