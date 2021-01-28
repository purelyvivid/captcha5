import torch
import torchvision
from torch import nn
import os

class autoencoder(nn.Module):
    def __init__(self, n_channel=1, n_class=36):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channel, 32, 5, stride=3, padding=2),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(32, 256, 5, stride=2, padding=5),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 32, 5, stride=2, padding=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, n_channel, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        self.classfier = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),  # b, 8, 3, 3
            nn.Flatten(),
            nn.Linear(576, 200),  
            nn.Linear(200, n_class),   
            
        )

    def forward(self, x):
        lant = self.encoder(x)
        rec = self.decoder(lant)
        c = self.classfier(lant)
        return rec, c
    
    def get_latent(self, x):
        return self.encoder(x)
    

