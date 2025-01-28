
import torch
from abc import ABC, abstractmethod
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw
import numpy as np
import random, sys


class FMNISTSampler():
    def __init__(self, args):

        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2*(x-0.5)),
            ]
        )
        
        fmnist = datasets.FashionMNIST("datasets/fmnist", train=True, download=True, transform = preprocess)

        self.dataloader = DataLoader(fmnist, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
        self.image_iterator = iter(self.dataloader)
        self.batch_size = args.batch_size
        self.length = len(self.dataloader)
        self.device = args.device


    def __iter__(self):
        return self

    def __next__(self):
        try:
            current_batch = next(self.image_iterator)
        except StopIteration:
            self.image_iterator = iter(self.dataloader)
            raise StopIteration
        [images, labels] = current_batch
        images = images.to(self.device)
        
        return images, labels
    

class MNISTSampler():
    def __init__(self, args):
        print(args)
        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (28,28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2*(x-0.5)),
            ]
        )
        
        mnist = datasets.MNIST("datasets/mnist", train=True, download=True, transform = preprocess)

        self.dataloader = DataLoader(mnist, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
        self.image_iterator = iter(self.dataloader)
        self.batch_size = args.batch_size
        self.length = len(self.dataloader)
        self.device = args.device


    def __iter__(self):
        return self

    def __next__(self):
        try:
            current_batch = next(self.image_iterator)
        except StopIteration:
            self.image_iterator = iter(self.dataloader)
            raise StopIteration
        [images, labels] = current_batch
        images = images.to(self.device)
        
        return images, labels
    



class ComboSampler():
    def __init__(self, args):

        preprocess_mnist = transforms.Compose(
            [
                transforms.Resize(
                    (28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2*(x-0.5)),
            ]
        )

        
        mnist = datasets.MNIST("datasets/mnist", train=True, download=True, transform = preprocess_mnist)
        fmnist = datasets.FashionMNIST("datasets/fmnist", train=True, download=True, transform = preprocess_mnist)


        self.mnist_percent = args.mnist_p
        if self.mnist_percent < 0.1:
            raise ValueError("Can't pass that small of a mnist_percent, need to ensure same dataset size")
        self.total_size = 66666

        base_batch_size = args.batch_size

        if self.mnist_percent < 0.5:
            self.mnist_batch_size = int(self.mnist_percent * base_batch_size / (1 - self.mnist_percent))
            self.fmnist_batch_size = base_batch_size
        else:
            self.mnist_batch_size = base_batch_size
            self.fmnist_batch_size = int((1 - self.mnist_percent) * base_batch_size / self.mnist_percent)
        
        print("BATCH SIZES", self.mnist_batch_size, self.fmnist_batch_size)
        self.mnist_dataloader = DataLoader(mnist, batch_size=self.mnist_batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
        self.mnist_image_iterator = iter(self.mnist_dataloader)
        self.fmnist_dataloader = DataLoader(fmnist, batch_size=self.fmnist_batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
        self.fmnist_image_iterator = iter(self.fmnist_dataloader)

        self.batch_size = args.batch_size

        self.length = self.total_size // (self.fmnist_batch_size + self.mnist_batch_size)
        self.device = args.device

        self.num_mnist = 0
        self.num_fmnist = 0


    def __iter__(self):
        return self

    def __next__(self):
        try:
            current_mnist_batch = next(self.mnist_image_iterator)
            current_fmnist_batch = next(self.fmnist_image_iterator)
            if self.num_mnist + self.num_fmnist > self.total_size:
                raise StopIteration
            if len(current_mnist_batch[0]) != self.mnist_batch_size or len(current_fmnist_batch[0]) != self.fmnist_batch_size:
                raise StopIteration
            self.num_mnist += len(current_mnist_batch[0])
            self.num_fmnist += len(current_fmnist_batch[0])
        except StopIteration:
            self.num_mnist = 0
            self.num_fmnist = 0
            self.mnist_image_iterator = iter(self.mnist_dataloader)
            self.fmnist_image_iterator = iter(self.fmnist_dataloader)
            raise StopIteration
        [mnist_images, mnist_labels] = current_mnist_batch
        mnist_images = mnist_images.to(self.device)
        [fmnist_images, fmnist_labels] = current_fmnist_batch
        fmnist_images = fmnist_images.to(self.device)

        # LABEL 0 IF MNIST, 1 IF FMNIST
        mnist_labels = torch.zeros(size=(len(mnist_labels),), dtype=torch.int32).to(self.device)
        fmnist_labels = torch.ones(size=(len(fmnist_labels),), dtype=torch.int32).to(self.device)

        images = torch.cat((mnist_images, fmnist_images), dim = 0)
        labels = torch.cat((mnist_labels, fmnist_labels), dim = 0)

        return images, labels
    
