import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torch import nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as tvt
import torchvision.ops as ops
import torchvision.utils as utils
import random
import skimage.io as io
import pickle
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

seed = 101
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
torch.autograd.set_detect_anomaly(True)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


class MyGrayscale(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        return torch.unsqueeze(torch.sum(img, axis=0) / 3, dim=0)
    
convert = tvt.Compose([tvt.CenterCrop((160, 160)), tvt.Resize((80, 80)), tvt.ToTensor(), MyGrayscale()])
data = torchvision.datasets.CelebA(root="./CelebA", transform=convert, download=True, split="all")


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(EncoderBlock, self).__init__()
        
        self.bn = bn
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1)
        self.leaky = nn.LeakyReLU(0.2, True)
        self.batchnorm = nn.BatchNorm2d(out_ch, 0.8)
    
    def forward(self, x):
        
        # Convolution, leaky ReLU, dropout and (optional) batch normalization.
        out = self.conv(x)
        if self.bn:
            out = self.batchnorm(out)
        out = self.leaky(out)
        # out = self.dropout(out)   

        
        return out
    

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(DecoderBlock, self).__init__()
        
        self.bn = bn
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.leaky = nn.LeakyReLU(0.2, True)
        self.batchnorm = nn.BatchNorm2d(out_ch, 0.8)
    
    def forward(self, x):
        
        # Convolution, leaky ReLU, dropout and (optional) batch normalization.
        out = self.conv(x)
        if self.bn:
            out = self.batchnorm(out)
        out = self.leaky(out)
        
        return out
    

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = [nn.Conv2d(1, 32, 7, padding=3), # 32 x 80 x 80
                        nn.BatchNorm2d(32, 0.8),
                        nn.LeakyReLU(0.2, True),
                        EncoderBlock(32, 64),
                        EncoderBlock(64, 128),
                        EncoderBlock(128, 256),
                        EncoderBlock(256, 512)] # 512 x 5 x 5
        self.encoder = nn.Sequential(*self.encoder)

        self.encoder_linear = nn.Linear(12800, 128)

        self.mu_head = [nn.Linear(128, 128),
                        nn.LeakyReLU(0.2, True)]
        self.mu_head = nn.Sequential(*self.mu_head)
        
        self.logvar_head = [nn.Linear(128, 128),
                        nn.LeakyReLU(0.2, True)]
        self.logvar_head = nn.Sequential(*self.logvar_head)

        self.decoder_linear = nn.Linear(128, 12800)

        self.decoder = [DecoderBlock(512, 256),
                        DecoderBlock(256, 128),
                        DecoderBlock(128, 64),
                        DecoderBlock(64, 32),
                        nn.Conv2d(32, 1, 7, padding=3),
                        nn.Sigmoid()]
        self.decoder = nn.Sequential(*self.decoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        encoding = self.encoder_linear(self.encoder(x).view(-1, 12800))

        mu = self.mu_head(encoding)
        logvar = self.logvar_head(encoding)

        latent = self.reparameterize(mu, logvar)

        reconstruction = self.decoder(self.decoder_linear(latent).view(-1, 512, 5, 5))

        return reconstruction, mu, logvar

    def sample(self, noise):
        with torch.no_grad():
            samples = self.decoder(self.decoder_linear(noise).view(-1, 512, 5, 5))
        return samples
    

def free_energy(input, reconstruction, mu, logvar):
    cross_entropy = F.binary_cross_entropy(reconstruction, input, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return cross_entropy + kl_divergence


def train_vae(model, train_dataloader, epochs, step_size):
        
    # Training routine    
    fixed_noise = torch.randn(8, 128, device=device)
    
    # Keep track of the losses for plotting later.
    losses = []
    img_list = []
    
    # Put both networks on GPU.
    model = model.to(device)
    
    print(f"Using device: {device}")
        
    # Use the Adam optimizer, with user specified step size.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .9))
    
    for epoch in range(epochs):
        # Keeping track of running loss.
        running_loss = 0.
        for i, item in enumerate(train_dataloader):
            item = item[0].to(device)

            reconstruction, mu, logvar = model(item)

            loss = free_energy(item, reconstruction, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            fake = model.sample(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake, padding=1, pad_value=1, normalize=True))

            running_loss += loss.item()

            if (i+1) % 50 == 0:
                if (i+1) % 100 == 0:
                    print("[epoch: %d, batch: %5d] loss: %.3f" % (epoch+1, i+1, running_loss / 50))
                losses.append(running_loss / 50)

            running_loss = 0.0
        
        # Plot our produced fakes at the end of each epoch. 
        plt.axis("off")                                                                                
        plt.title("Fake Images")                                                                       
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))

        plt.savefig("vae_fake_images.png")                                 
        plt.show()    
        
        # Save the model and the loss values every 5 epochs.
        if epoch % 5 == 4:
            torch.save(model.state_dict(), f"models/vae_{epoch+41}")
        
            with open(f"losses/vae_losses_{epoch+41}", "wb") as fp:
                pickle.dump(losses, fp)
                
    return losses


split1, _, _ = torch.utils.data.random_split(data, [100000, 100000, 2599])


if __name__ == "__main__":
    train_dataloader = torch.utils.data.DataLoader(split1, batch_size=32, num_workers=8, shuffle=True, drop_last=True)

    model = VAE()

    model.load_state_dict(torch.load("models/vae_40"))

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Number of parameters (millions):", model_params / 1e6)
    epochs = 200

    losses = train_vae(model, train_dataloader, epochs, 1e-4)