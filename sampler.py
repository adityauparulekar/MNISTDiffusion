
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

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
        
        indexed_fmnist = IndexedDataset(dataset="fmnist", train=True, transform=preprocess)

        self.dataloader = DataLoader(indexed_fmnist, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
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
        [images, labels, indices] = current_batch
        images = images.to(self.device)
        
        return images, labels, indices

class CIFARSampler():
    def __init__(self, args):

        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2*(x-0.5)),
                transforms.Grayscale(num_output_channels=1)
            ]
        )
        
        indexed_fmnist = IndexedDataset(dataset="cifar", train=True, transform=preprocess)

        self.dataloader = DataLoader(indexed_fmnist, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
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
        [images, labels, indices] = current_batch
        images = images.to(self.device)
        
        return images, labels, indices

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
        
        indexed_mnist = IndexedDataset(dataset="mnist", train=True, transform=preprocess)

        self.dataloader = DataLoader(indexed_mnist, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
        self.image_iterator = iter(self.dataloader)
        self.batch_size = args.batch_size
        self.length = len(self.dataloader)
        self.device = args.device
        self.job_id = args.job_id
        self.num_jobs = args.num_jobs

    def __iter__(self):
        return self

    def __next__(self):
        try:
            current_batch = next(self.image_iterator)
        except StopIteration:
            self.image_iterator = iter(self.dataloader)
            raise StopIteration
        [images, labels, indices] = current_batch
        images = images.to(self.device)
        total_batches = len(images) // self.num_jobs
        start_idx = self.job_id * total_batches
        end_idx = (self.job_id + 1) * total_batches if self.job_id < self.num_jobs - 1 else self.length
        return images[start_idx:end_idx], labels[start_idx:end_idx], indices[start_idx:end_idx]
    



class ComboSampler():
    def __init__(self, args):

        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2*(x-0.5)),
            ]
        )
        
        indexed_mnist = IndexedDataset(dataset="mnist", train=args.train, transform=preprocess)
        indexed_fmnist = IndexedDataset(dataset="fmnist", train=args.train, transform=preprocess)

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
        self.mnist_dataloader = DataLoader(indexed_mnist, batch_size=self.mnist_batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
        self.mnist_image_iterator = iter(self.mnist_dataloader)
        self.fmnist_dataloader = DataLoader(indexed_fmnist, batch_size=self.fmnist_batch_size, shuffle=True, generator=torch.Generator(device='cpu'))    
        self.fmnist_image_iterator = iter(self.fmnist_dataloader)

        self.correct_batches = args.lr_correction
        if self.correct_batches:
            print("I NEVER GOT IN HERE")
            df = pd.read_csv('errors.csv')
            self.mnist_errors = df.groupby(['image_idx', 'dataset']).mean()['error'][:, 0]
            self.fmnist_errors = df.groupby(['image_idx', 'dataset']).mean()['error'][:, 1]
            self.mnist_errors /= self.mnist_errors.sum()
            self.fmnist_errors /= self.fmnist_errors.sum()

            self.mult = int(min(1 / self.mnist_errors.max(), 1 / self.fmnist_errors.max()) - 1)
            print("MULT", self.mult)
        self.batch_size = args.batch_size

        self.length = self.total_size // (self.fmnist_batch_size + self.mnist_batch_size)
        self.device = args.device

        self.num_mnist = 0
        self.num_fmnist = 0

    def resample_batch(self, b, p, M):
        images, labels, indices = b
        new_i = []
        for i, ind in enumerate(indices):
            if np.random.random() < p[ind.item()] * M:
                new_i.append(i)
        new_images = images[new_i]
        new_labels = labels[new_i]
        new_indices = indices[new_i]
        return (new_images, new_labels, new_indices)
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            current_mnist_batch = next(self.mnist_image_iterator)
            current_fmnist_batch = next(self.fmnist_image_iterator)
            if len(current_mnist_batch[0]) != self.mnist_batch_size or len(current_fmnist_batch[0]) != self.fmnist_batch_size:
                raise StopIteration
            if self.num_mnist + self.num_fmnist > self.total_size:
                raise StopIteration
            self.num_mnist += len(current_mnist_batch[0])
            self.num_fmnist += len(current_fmnist_batch[0])
            if self.correct_batches:
                print("NEVER GOT IN HERE EITHER")
                current_mnist_batch = self.resample_batch(current_mnist_batch, self.mnist_errors, self.mult)
                current_fmnist_batch = self.resample_batch(current_fmnist_batch, self.fmnist_errors, self.mult)
        except StopIteration:
            self.num_mnist = 0
            self.num_fmnist = 0
            self.mnist_image_iterator = iter(self.mnist_dataloader)
            self.fmnist_image_iterator = iter(self.fmnist_dataloader)
            raise StopIteration

        [mnist_images, mnist_labels, mnist_indices] = current_mnist_batch
        mnist_images = mnist_images.to(self.device)
        [fmnist_images, fmnist_labels, fmnist_indices] = current_fmnist_batch
        fmnist_images = fmnist_images.to(self.device)

        # LABEL 0 IF MNIST, 1 IF FMNIST
        mnist_labels = mnist_labels * 0
        fmnist_labels = fmnist_labels * 0 + 1

        images = torch.cat((mnist_images, fmnist_images), dim = 0)
        labels = torch.cat((mnist_labels, fmnist_labels), dim = 0)
        indices = torch.cat((mnist_indices, fmnist_indices), dim=0)

        return images, labels, indices

class GaussianSampler():
    def compute_weights(x):
        D = x.shape[-1]
        norm_sq = (x ** 2).sum(dim=(2, 3))
        result = (norm_sq ** 2 + (2 * D + 4) * norm_sq + D * (D + 2)) / 16.0
        return result

    def rejection_sample(data):
        weights = GaussianSampler.compute_weights(data)
        normalized_weights = (weights / torch.sum(weights)).squeeze(-1)
        resampled_indices = torch.multinomial(normalized_weights, num_samples=data.shape[0], replacement=True)
        return data[resampled_indices]
    
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dataset_size = 10000
        self.length = self.dataset_size // self.batch_size
        self.device = args.device
        self.num_emitted = 0

        gaussian_data = torch.normal(0, 1, size=(self.dataset_size, 1, args.image_size, args.image_size))
        if args.reweight:
            self.data = GaussianSampler.rejection_sample(gaussian_data)
        else:
            self.data = gaussian_data

    def __iter__(self):
        self.num_emitted = 0
        return self

    def __next__(self):
        if self.num_emitted + self.batch_size > self.dataset_size:
            self.num_emitted = 0
            raise StopIteration
        current_batch = self.data[self.num_emitted: self.num_emitted+self.batch_size]
        self.num_emitted += self.batch_size
        labels = torch.zeros((self.batch_size,))
        indices = torch.zeros((self.batch_size,))
        return (current_batch, labels, indices)

class IndexedDataset(Dataset):
    def __init__(self, dataset, train=True, download=True, transform=None):
        if dataset == "mnist":
            self.data = datasets.MNIST("datasets/mnist", train=train, download=True, transform=transform)
        elif dataset == "fmnist":
            self.data = datasets.FashionMNIST("datasets/fmnist", train=train, download=True, transform=transform)
        elif dataset == "cifar":
            self.data = datasets.CIFAR10('datasets/cifar', train=train, download=True, transform=transform)
        else:
            raise ValueError("Please pass valid dataset mnist or fmnist")
        self.indices = list(range(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label, self.indices[idx]

