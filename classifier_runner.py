# %%
# Imports

# Pytorch
import torch

import sys
sys.path.append("..")
# HuggingFace
import datasets
import yaml

# Training and Visualization
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import PIL

# Causal Diffusion Model
import classifier
from sampler import MNISTSampler, ComboSampler
from dataclasses import dataclass

import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config.yaml', type=str)
    parser.add_argument('--mnist_p', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_file', type=str, default='models/classifier.pth')
    parser.add_argument('--checkpoint', type=str, default='models/classifier.pth')
    args = parser.parse_args()
    
    torch.set_default_device(args.device)
    
    sampler = ComboSampler(args)
    Classifier = classifier.Classifier(args)
    Classifier.train_loop(sampler)
