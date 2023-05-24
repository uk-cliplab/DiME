import torch
import torch.nn as nn
import numpy as np
from math import prod

class Big_Convolutional_Encoder(nn.Module):
    def __init__(self, latent_dim, linear_size = 1024, img_size=(1, 28, 28)):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(img_size[0], 32, kernel_size=3, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten()
            
        self.encoder_lin = nn.Sequential(
            nn.LazyLinear(out_features=linear_size),
            nn.ReLU(True),
            nn.Linear(linear_size, latent_dim),
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Big_Convolutional_Decoder2828(nn.Module):
    def __init__(self, latent_dim, linear_size=1024, img_size=(1, 28, 28)):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, linear_size),
            nn.ReLU(True),
            nn.Linear(linear_size, 5*5*128),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(128, 5,5))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, img_size[0], 3, stride=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class Convolutional_Encoder(nn.Module):
    def __init__(self, latent_dim, linear_size, img_size=(1, 28, 28)):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(img_size[0], 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.LazyLinear(linear_size),
            nn.ReLU(True),
            nn.Linear(linear_size, latent_dim),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Convolutional_Decoder_2828(nn.Module):
    """
    Outputs 28x28 images
    """
    def __init__(self, latent_dim, linear_size, img_size=(1, 28, 28)):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, linear_size),
            nn.ReLU(True),
            nn.Linear(linear_size, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, img_size[0], 3, stride=2, 
            padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class Simple_Classifier(nn.Module):
    """
    Implementation of simple classifier
    """
    def __init__(self, latent_dim, num_classes=10):
        super().__init__()

        ### Linear section
        self.lin = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.Linear(1024, num_classes),
        )
        
    def forward(self, x):
        x = self.lin(x)
        return x
    
class Multiview_Autoencoder(nn.Module):
    """
    Multiview autoencoder. Has several features

    Inputs:
        latent_dim - size of latent_space for MI calcuations
        recon_latent_dim - size of latent_space untouched by MI calculations\
                NOTE - total latent space size is latent_dim + recon_latent_dim

        num_views - number of views. only 2 is supported right now
        linear_size - linear size of encoders and decoders
        architecture - whether to use MLP or CNN

    By default, all models of this type have 2 encoders, 2 decoders, 1 classifier, and learnable sigmas.
    The loss functions dictate which of these actually are important.
    For example, if no reconstruction loss is used, then the decoders won't learn anything or contribute to learning.
    TODO - make models tailor-made to increase efficiency.
    """

    def __init__(self, latent_dim, recon_latent_dim=0, num_views=2, linear_size=1024, num_classes=10, img_size=(1,28,28), architecture="MLP", single_encoder=False, is_stacked=False, is_audio=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.recon_latent_dim = recon_latent_dim
        self.total_latent_size = latent_dim + recon_latent_dim
        self.single_encoder = single_encoder
        self.is_stacked = is_stacked

        if architecture == "CNN":
            encoder_arch, decoder_arch = Convolutional_Encoder, Convolutional_Decoder_2828
        elif architecture == "bigCNN":
            encoder_arch, decoder_arch = Big_Convolutional_Encoder, Big_Convolutional_Decoder2828
        else:
            raise ValueError("INVALID ARCHITECTURE TYPE")

        self.encoder_view0 = encoder_arch(self.total_latent_size, linear_size=linear_size, img_size=img_size)
        self.decoder_view0 = decoder_arch(self.total_latent_size, linear_size=linear_size, img_size=img_size)

        self.encoder_view1 = encoder_arch(self.total_latent_size, linear_size=linear_size, img_size=img_size)
        self.decoder_view1 = decoder_arch(self.total_latent_size, linear_size=linear_size, img_size=img_size)
    
        self.classifier_view0 = Simple_Classifier(self.total_latent_size, num_classes=num_classes)
        self.classifier_view1 = Simple_Classifier(self.total_latent_size, num_classes=num_classes)

        self.num_views = num_views
        self.cur_view = 0
        
        self.sigma_x = torch.nn.Parameter(torch.tensor(1.))
        self.sigma_y = torch.nn.Parameter(torch.tensor(1.))

    def change_view(self, cur_view):
        """
        Change the model view to determine which encoder/decoder to pass through
        """
        if cur_view not in list(range(self.num_views)):
            raise ValueError("Invalid view number!")

        self.cur_view = cur_view

    def forward(self, x):
        if self.cur_view == 0:
            enc = self.encoder_view0(x)
            dec = self.decoder_view0(enc)
            classifications = self.classifier_view0(enc)

        elif self.cur_view == 1:
            enc = self.encoder_view1(x)
            dec = self.decoder_view1(enc)
            classifications = self.classifier_view1(enc)
    
        # returns MI dims, reconstruction, all latent dims, classifier results
        return enc[:, 0:self.latent_dim], dec, enc, classifications

    def get_whole_latent(self, x):
        """
        Get all latent dims. Equivalent to running forward() and taking the third output
        """
        if self.cur_view == 0:
            enc = self.encoder_view0(x)
        elif self.cur_view == 1:
            enc = self.encoder_view1(x)

        return enc

    def decode_modified_latent(self, enc):
        """
        Get a reconstruction for a passed-in latent embedding.
        """
        if self.cur_view == 0:
            dec = self.decoder_view0(enc)
        elif self.cur_view == 1:
            dec = self.decoder_view1(enc)
        return dec