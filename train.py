import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from tqdm import tqdm
from torchvision.utils import save_image
from sampler import FMNISTSampler, MNISTSampler, CIFARSampler, GaussianSampler
from sampler import ComboSampler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os, sys
import math
import argparse
import pandas as pd

def norm_sum(a, b):
    return (a - b)**2 / sum((a-b)**2)

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
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
    parser.add_argument('--lr_correction', action='store_true')
    parser.add_argument('--bias_correction', action='store_true')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--store_errors', action='store_true')
    args = parser.parse_args()

    return args


def main(args):
    x = 1
    log_file = open('log_file.txt', 'w')
    s = str(args)
    print(s)
    log_file.write(s)
    log_file.write("\n")
    log_file.close()
    
    device = args.device
    if args.dataset == 'gaussian':
        train_dataloader = GaussianSampler(args)
        model=MNISTDiffusion(timesteps=args.timesteps,
                    image_size=2,
                    in_channels=1,
                    base_dim=8,
                    dim_mults=[2,4],
                    small_input=True).to(device)
    else:
        if args.dataset == 'mnist':
            train_dataloader = MNISTSampler(args)
        elif args.dataset == 'fmnist':
            train_dataloader = FMNISTSampler(args)
        elif args.dataset == 'cifar':
            train_dataloader = CIFARSampler(args)
        elif args.dataset == 'combo':
            train_dataloader = ComboSampler(args)
        model=MNISTDiffusion(timesteps=args.timesteps,
                    image_size=28,
                    in_channels=1,
                    base_dim=args.model_base_dim,
                    dim_mults=[2,4]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*train_dataloader.length*2,pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps=0
    errors_list = []

    if args.store_errors or args.lr_correction:
        df = pd.read_csv("errors.csv")
        avg_error = df['error'].mean()
        df_errors = df.groupby(['image_idx', 'dataset']).mean()['error']
    for i in range(args.epochs):
        progress_bar = tqdm(total=train_dataloader.length)
        progress_bar.set_description(f"Epoch {i}")
        model_ema.train()
        model.train()
        for j,(images, labels, indices) in enumerate(train_dataloader):
            noise=torch.randn_like(images).to(device)
            images=images.to(device)
            pred=model(images,noise)
            if args.lr_correction:
                print("SHOULDNT BE HERE EITHER")
                error_inds = torch.stack((indices, labels), dim=1).tolist()
                errors = df_errors.groupby(['image_idx', 'dataset']).mean()[error_inds].tolist()
                upweights = avg_error / errors
                for ind in range(len(noise)):
                    noise[ind] *= math.sqrt(upweights[ind])
                    pred[ind] *= math.sqrt(upweights[ind])
            elif args.bias_correction:
                print("OR HERE")
                true_bias = [0.5, 0.5]
                alt_bias = [args.mnist_p, 1.0-args.mnist_p]
                upweights = [true_bias[l] / alt_bias[l] for l in labels]
                for ind in range(len(noise)):
                    noise[ind] *= math.sqrt(upweights[ind])
                    pred[ind] *= math.sqrt(upweights[ind])
            loss=loss_fn(pred,noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            ### TRACKING STUFF, NO TRAINING LOGIC BELOW ###
            progress_bar.update(1)
            logs = {
                "loss" : loss.detach().item(),
                "lr" : scheduler.get_last_lr()[0],
                # "mnist": train_dataloader.num_mnist,
                # "fmnist": train_dataloader.num_fmnist,
            }
            progress_bar.set_postfix(**logs)
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            
            if args.store_errors and i % 10 == 0:
                for ind in range(len(images)):
                    image_rep = images[ind].repeat((100, 1, 1, 1))
                    noise=torch.randn_like(image_rep).to(device)
                    pred=model(image_rep,noise)
                    diff = pred - noise
                    x = torch.mean(diff**4).item()
                    errors_list.append({
                        'epoch': i,
                        'image_idx': indices[ind].item(),
                        'label': labels[ind].item(),
                        'error': x
                    })
            ### TRACKING STUFF, NO TRAINING LOGIC ABOVE ###
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}

        os.makedirs("models",exist_ok=True)
        torch.save(ckpt,f"models/{args.name}.pt")

        if i % 5 == 0:
            model_ema.eval()
            samples = sample(model_ema, args, device)
            print("DISPLAYING IMAGE")
            os.makedirs("images",exist_ok=True)
            save_image(samples,f"images/{args.name}.png",nrow=int(math.sqrt(args.n_samples)))
    if args.store_errors:
        df = pd.DataFrame(errors_list)
        df.to_csv('fmnist_weights.csv', index=False)
def sample(model_ema, args, device, seed=2025):
    torch.manual_seed(seed)
    return model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)

if __name__=="__main__":
    args=parse_args()
    main(args)