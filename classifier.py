import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from classifier_net import Net
from tqdm import tqdm
import diffusers

class Classifier():
    def __init(self, args):
        self.save_file = args["save_file"]
        
        self.device = torch.device(args.device)

        self.net = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=400)

        self.num_epochs = args.num_epochs
        self.save_file = args.save_file
        self.checkpoint = args.checkpoint
    def train_loop(self, trainloader):
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            progress_bar = tqdm(total=trainloader.length)
            progress_bar.set_description(f"Epoch {epoch}")

            running_loss = 0.0
            for i, (inputs, labels, _) in enumerate(trainloader, 0):
                # zero the parameter gradients

                self.optimizer.zero_grad()
                clean_images = inputs.to(self.device)
                batch_size = clean_images.shape[0]

                noise = torch.randn(inputs.shape).to(self.device)
                timesteps = torch.ones((batch_size,), device=self.device, dtype=torch.long)
                noisy_images = self.noise_scheduler.add_noise(inputs, noise, timesteps)
                # forward + backward + optimize
                outputs = self.net(noisy_images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                progress_bar.update(1)
                logs = {
                    "loss" : loss.detach().item(),
                }
                progress_bar.set_postfix(**logs)
                self.save_model(self.save_file)
    
    # LOAD AND SAVE
    def save_model(self, PATH):
        if os.path.exists(PATH):
            os.remove(PATH)
        state = {}
        state.update({'net': self.net.state_dict()})
        torch.save(state, PATH)
    
    def load_model(self, PATH):
        PATH = self.checkpoint
        checkpoint = torch.load(PATH)
        self.net.load_state_dict(checkpoint['net'])
        self.net.train()