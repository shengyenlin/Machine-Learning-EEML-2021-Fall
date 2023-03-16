import csv
import time
import sys
import os
import random
import cv2
import copy

# other library
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImgDataset(Dataset):
    def __init__(self, data, transform):
        self.total_img = data
        self.total_img = self.total_img.permute(0, 3, 1, 2) #(N, C, W, H)
        self.transform = transform
        
    def normalize(self, img):
        #img = img / 255
        return img
    
    def augment(self, img):
        # TODO
        return img

    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, index):
        img = self.total_img[index]
        img = self.transform(img)
        return img

class VAE(nn.Module):
    def __init__(self, imgChannels, feature_dim, latent_dim):
        super(VAE, self).__init__()
        self.feature_dim = feature_dim
        self.conv_stage = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.encMuFC = nn.Linear(feature_dim, latent_dim)
        self.encVarFC = nn.Linear(feature_dim, latent_dim)
        
        self.decFC = nn.Linear(latent_dim, feature_dim)
        self.trans_conv_stage = nn.Sequential(
            
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, padding=1,bias=False)
        )

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.conv_stage(x)
        #print(x.shape)
        x = x.view(-1, self.feature_dim)
        #print(x.shape)
        mu = self.encMuFC(x)
        logVar = self.encVarFC(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        # std = torch.exp(logVar/2)
        # eps = torch.randn_like(std)
        std = logVar.mul(0.5).exp_() #multiply each element 
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.cuda()
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.decFC(z)
        x = x.view(-1, 256, 2, 2)
        x = self.trans_conv_stage(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

class AE(nn.Module):
    def __init__(self, imgChannels, feature_dim, latent_dim):
        super(AE, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        #in: (3, 32, 32)
        self.conv_stage = nn.Sequential(
            #16 x 16 x 32
            nn.Conv2d(
                in_channels = imgChannels, out_channels = 32, 
                kernel_size = 4, stride = 2, padding = 1
                ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),

            #8 x 8 x 64 
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            #4 x 4 x 128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),

            #2 x 2 x 256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.enc_fc = nn.Linear(feature_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, feature_dim)

        self.trans_conv_stage = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, padding=1,bias=False)
        )

    def encoder(self, x):
        x = self.conv_stage(x)
        x = x.view(-1, self.feature_dim)
        x = self.enc_fc(x)
        return x

    def decoder(self, x):
        x = self.dec_fc(x)
        x = x.view(-1, 256, 2, 2)
        x = self.trans_conv_stage(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x_latent = self.encoder(x)
        x_out = self.decoder(x_latent)
        return x_out, x_latent

def change_to_submit_format_df(result):
    ids = np.arange(9000)
    submit_dict = {
        "id": ids,
        "label": result
    }
    submit_df = pd.DataFrame(submit_dict)
    return submit_df

def output_to_csv(df, submit_path):
    df.to_csv(submit_path, index =False)

def loss_fn(pred, true):
    loss = nn.L1Loss(reduction='sum')
    l1_loss = loss(pred, true)
    return l1_loss

def training(train_loader, model, device, optimizer, n_epoch):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=== start training, parameter total:%d, trainable:%d' % (total, trainable))
                        
    best_loss = np.inf
    best_epoch = 0
    best_model = None
    for epoch in range(n_epoch):
        total_loss = 0
        model.train()
        for idx, img in enumerate(train_loader):
            img = img.to(device, dtype=torch.float)
            img_out, img_latent = model(img)
            loss = loss_fn(img_out, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        total_loss /= len(train_loader.dataset)
        print("Epoch {}  | Loss:{:.4f}".format(epoch+1, total_loss))
        if total_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = total_loss
            best_epoch = epoch
    print("Best Epoch is {}  | Loss:{:.4f}".format(best_epoch+1, best_loss))
    return best_model