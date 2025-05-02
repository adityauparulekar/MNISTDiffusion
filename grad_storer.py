import torch
import torch.nn as nn
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from PIL import Image
from classifier import Classifier
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import argparse
from sampler import MNISTSampler, FMNISTSampler
from torch.utils.data import Dataset
import sys
from torchvision import datasets, transforms

# args = Namespace(batch_size=128, model_ema_steps=10, epochs=100, model_ema_decay=0.995, timesteps=1000, model_base_dim=64, mnist_p=0.5, lr_correction=False, device='cuda:1', train=False)
parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
parser.add_argument('--lr',type = float ,default=0.001)
parser.add_argument('--batch_size',type = int ,default=128)    
parser.add_argument('--epochs',type = int,default=100)
parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
parser.add_argument('--device', type=str, help='device to run on, either cuda:0-7 or cpu', default='cuda:0')
parser.add_argument('--name', type=str, help='name of the experiment', default='exp')
parser.add_argument('--mnist_p', type=float, help='percent of mnist in dataset', default=0.5)
parser.add_argument('--grad_correction', action='store_true')
parser.add_argument('--z_reps', type=int, default=1)
parser.add_argument('--model_file', type=str)
parser.add_argument('--grad_file', type=str)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
parser.add_argument('--hess', action='store_true')
parser.add_argument('--store_grads', action='store_true')
parser.add_argument("--job_id",   type=int, required=True, help="0-based index of this SLURM array task")
parser.add_argument("--num_jobs", type=int, required=True, help="total number of SLURM array tasks")
args = parser.parse_args()

adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
alpha = 1.0 - args.model_ema_decay
alpha = min(1.0, alpha * adjust)
device = args.device

model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)
model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
ckpt=torch.load(args.model_file)
model_ema.load_state_dict(ckpt['model_ema'])
model.load_state_dict(ckpt['model'])
sampler = MNISTSampler(args)

loss_fn = nn.MSELoss(reduction='mean')
optimizer=AdamW(model.parameters(),lr=0.001)

def approx_hessian_diag(model, loss, n_probes=10):
    params = [p for p in model.parameters() if p.requires_grad]
    diag_est = [torch.zeros_like(p) for p in params]

    for _ in tqdm(range(n_probes)):
        vs = [torch.randint_like(p, low=0, high=2, dtype=torch.float32) * 2 - 1
              for p in params]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        hvp = torch.autograd.grad(
            grads, params, grad_outputs=vs, retain_graph=True
        )
        for i, (v, hv) in enumerate(zip(vs, hvp)):
            diag_est[i] += v * hv
        torch.cuda.empty_cache()
    for i in range(len(diag_est)):
        diag_est[i] /= n_probes

    return diag_est

loss = 0
count = 0
if args.hess:
    for i, (images, labels, indices) in enumerate(sampler):
        count += 1
        if count > 2:
            break
        noise = torch.randn_like(images).to(device)
        pred = model(images, noise)

        loss += loss_fn(pred, noise)
    print("approximating hessian")
    hess = approx_hessian_diag(model, loss, n_probes=100)
print("calculating grads")

f = open('grads/' + args.grad_file + str(args.job_id) + '.txt', 'a')
for i, (images, labels, indices) in enumerate(sampler):
    images=images.to(device)    
    curr_batch = []
    for ind in tqdm(range(len(images))):
        image = images[ind:ind+1].to(device)
        S = 100
        total_grad_norm = 0.0
        for j in range(S):
            timesteps = torch.ones((len(image),),dtype=torch.int64).to(device)
            time_mult = 0
            model.zero_grad()

            noise = torch.randn_like(image).to(device)

            pred = model(image, noise, t=timesteps*time_mult)
            loss = loss_fn(pred, noise)
            loss.backward()

            total = 0.0
            if args.hess:
                for param, h in zip(model.parameters(), hess):
                    if param.grad is not None:
                        total += (param.grad/h.clip(min=0.01)).norm()**2
            else:
                for param in model.parameters():
                    total += param.grad.norm()**2
            total_grad_norm += np.sqrt(total.item())
        total_grad_norm /= S
        curr_batch.append({
            'image_idx': indices[ind].item(),
            'label': labels[ind].item(),
            'grad_norm': total_grad_norm.item(),
        })
        optimizer.zero_grad()
    f.write(str(curr_batch) + "\n")
    f.flush()
    torch.cuda.empty_cache()
f.close()
