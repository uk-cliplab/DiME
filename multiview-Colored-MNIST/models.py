import torch
import torch.nn as nn

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 32, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, padding=1),
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Linear(7*7*64, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, z_dim)
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
    
class Decoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 7*7*64),
            torch.nn.ReLU(),
            Reshape((64,7,7,)),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, z_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


class BigOld_Convolutional_Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            # nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            # nn.BatchNorm2d(256),
            nn.Flatten(),
            
            nn.Linear(144*4*4, 1024),
            # torch.nn.ReLU(),
            torch.nn.Linear(1024, latent_dim),
            
        )
        
    def forward(self, xb):
        return self.network(xb)

class BigOld_Convolutional_Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024,256*4*4),
            nn.ReLU(),

            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            # Reshape((256,3,3,)),

            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, xb):
        return self.network(xb)


class BigOldModel(torch.nn.Module):
    def __init__(self, z_dim):
        super(BigOldModel, self).__init__()
        self.encoder = BigOld_Convolutional_Encoder(latent_dim = z_dim)
        self.decoder = BigOld_Convolutional_Decoder(latent_dim = z_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed

class Big_Convolutional_Encoder(nn.Module):
    def __init__(self, latent_dim, linear_size = 1024, img_size=(3, 28, 28)):
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

class Big_Convolutional_Decoder(nn.Module):
    def __init__(self, latent_dim, linear_size=1024, img_size=(3, 28, 28)):
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
            nn.ReLU(),

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

class BigModel(torch.nn.Module):
    def __init__(self, latent_dim, linear_size=1024, img_size=(3, 28, 28)):
        super(BigModel, self).__init__()
        self.encoder = Big_Convolutional_Encoder(latent_dim = latent_dim, linear_size=linear_size, img_size=img_size)
        self.decoder = Big_Convolutional_Decoder(latent_dim = latent_dim, linear_size=linear_size, img_size=img_size)
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


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