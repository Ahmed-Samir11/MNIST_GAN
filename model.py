import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = latent_dim + num_classes
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((z, label_input), dim=1)
        img_flat = self.model(x)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = int(torch.prod(torch.tensor(img_shape))) + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_emb(labels)
        x = torch.cat((img_flat, label_input), dim=1)
        return self.model(x)