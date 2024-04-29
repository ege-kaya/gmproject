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

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


class GANBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(GANBlock, self).__init__()
        
        self.bn = bn
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1)
        self.leaky = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout2d(0.25)
        self.batchnorm = nn.BatchNorm2d(out_ch, 0.8)
    
    def forward(self, x):
        
        # Convolution, leaky ReLU, dropout and (optional) batch normalization.
        out = self.conv(x)
        if self.bn:
            out = self.batchnorm(out)
        out = self.leaky(out)
        # out = self.dropout(out)   

        
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
                
        model = [nn.ConvTranspose2d(100, 1024, 3, stride=1, padding=0), #512 x 3 x 3
                 nn.BatchNorm2d(1024),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=0), #256 x 5 x 5
                 nn.BatchNorm2d(512),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0), #128 x 7 x 7
                 nn.BatchNorm2d(256),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), #128 x 14 x 14
                 nn.BatchNorm2d(128),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1), #1 x 28 x 28
                 nn.Tanh()
                 ]
        
        self.model = nn.Sequential(*model)
            
    def forward(self, x):
        out = self.model(x)

        return out
    

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        model = [nn.Conv2d(1, 128, 7, padding=3),
                nn.BatchNorm2d(128, 0.8), # 32x28x28
                 nn.LeakyReLU(0.2, True), 
                #  nn.Dropout2d(0.25), 
                 GANBlock(128, 256),       # 64x14x14
                 GANBlock(256, 512),      # 128x7x7
                 GANBlock(512, 1024)]      # 256x4x4
        
        # Here is one difference from the Discriminator network: the Sigmoid layer is gone!
        self.linear = nn.Linear(16384, 1)
        
        self.model = nn.Sequential(*model)
    
    # The forward method simply takes the inputs through the network explained above.
    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        
        # And here is the second difference: we take the mean of the values over the batch,
        # and return this value as the output of the network.
        out = self.linear(out).mean(axis=0)
        
        return out
    

def gradient_penalty(batch_size, critic, reals, fakes, Lambda=10):
    """
    Implementation by Marvin Cao: https://github.com/caogang/wgan-gp
    Marvin Cao's code is a PyTorch version of the Tensorflow based implementation provided by
    the authors of the paper "Improved Training of Wasserstein GANs" by Gulrajani, Ahmed, 
    Arjovsky, Dumouli,  and Courville. Code taken from Prof. Avi Kak's DLStudio.
    """
    epsilon = torch.rand(1).to(device)
    interpolates = epsilon * reals + ((1 - epsilon) * fakes)
    interpolates = interpolates.requires_grad_(True).to(device) 
    interpolates_out = critic(interpolates)
    gradients = torch.autograd.grad(outputs=interpolates_out, inputs=interpolates,
                              grad_outputs=torch.ones(interpolates_out.size()).to(device), 
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    
    return gp


def train_wgan(critic, generator, train_dataloader, epochs, step_size):
        
    # Training routine    
    fixed_noise = torch.randn(8, 100, 1, 1, device=device)
    one = torch.FloatTensor([1]).to(device)
    minus_one = torch.FloatTensor([-1]).to(device)
    
    # Keep track of the losses for plotting later.
    c_losses = []
    g_losses = []
    img_list = []
    
    # Put both networks on GPU.
    critic = critic.to(device)
    generator = generator.to(device)
    
    # With a WGAN, it might be a good idea to train the Critic network more than the Generator
    # network. This number essentially sets how much more we will train the Critic than the
    # Generator. Here, we have set it to 2, so the Critic gets trained twice as much.
    ncritic = 5
    
    print(f"Using device: {device}")
        
    # Use the Adam optimizer, with user specified step size.
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(.5, .9))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(.5, .9))
    
    for epoch in range(epochs):
        
        # For the Critic-getting-trained-more-than-Generator mechanic mentioned above, we
        # manually operate the iterator, instead of using it through a for loop.
        data_iter = iter(train_dataloader)
        i = 0
        
        # Keeping track of running losses.
        running_loss_g = 0.
        running_loss_c = 0.
        
        while i < len(train_dataloader):
            
            # Turn the Critic training on, because it is turned off further below.
            for p in critic.parameters():
                p.requires_grad = True  
            
            # To ensure more intensive training of the Critic.
            ic = 0
            while ic < ncritic and i < len(train_dataloader):
                ic += 1
                critic.zero_grad()
                
                # Get the real images.
                reals, _ =  next(data_iter)
                i += 1
                batch_size = reals.shape[0]
                reals = reals.to(device)
                
                # Get the output of the Critic with respect to these real images.
                reals_out = critic(reals)
                
                # Here's something weird: we set a gradient target! We set it to -1 for
                # real images.
                reals_out.backward(minus_one)
                
                # Now, we create a batch of fake images using our Generator.
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fakes = generator(noise)
                
                # Just like in the DCGAN training, we want the output of our Critic with
                # respect to these FAKE images. We make sure to detach them since we don't
                # want to update the Generator right now.
                fakes_out = critic(fakes.detach())
                
                # And here, we set out gradient target to 1.
                fakes_out.backward(one)
                
                # Now, for the final component, we calculate the gradient penalty, with the
                # function we declared beforehand. Here, we use a lambda value of 10.
                gp = gradient_penalty(batch_size, critic, reals, fakes, 10)
                gp.backward()               
                
                # The loss value for the Critic is the loss due to fakes minus the loss due
                # to reals, plus the gradient penalty value.
                loss_c = fakes_out - reals_out

                #  Update the critic.
                optimizer_c.step()   

                with torch.no_grad():             
                    fake = generator(fixed_noise).detach().cpu()

                img_list.append(utils.make_grid(fake, padding=1, pad_value=1, normalize=True))

                if (i+1) % 50 == 0:
                    if (i+1) % 100 == 0:
                        print("[epoch: %d, batch: %5d] crit. loss: %.3f" % (epoch+1, i+1, running_loss_c / 50))
                        print("[epoch: %d, batch: %5d] gen. loss: %.3f" % (epoch+1, i+1, running_loss_g / 50))
                    c_losses.append(running_loss_c / 50)
                    g_losses.append(running_loss_g / 50)

                    running_loss_c = 0.0
                    running_loss_g = 0.0
            # Update the running losses.
            running_loss_c += loss_c.item()
      
            # Turn off Critic training, we are done with it for now.
            for p in critic.parameters():
                p.requires_grad = False
            
            # Now it's time to train the Generator.
            generator.zero_grad()                         
            # Create a new batch of fake images and get their output from the Critic.
            noise = torch.randn(batch_size, 100, 1, 1, device=device)    
            fakes = generator(noise)          
            fakes_out = critic(fakes)
            
            # The Generator loss is simply the feedback of the Critic. How fake does it think
            # our images are?
            loss_g = fakes_out
            
            # Now, the gradient target of our Generator will also be -1.
            loss_g.backward(minus_one)      
            
            #  Update the Generator.
            optimizer_g.step()                                                                          
            
            # Update the running loss.
            running_loss_g -= loss_g.item()

            # Just some user feedback reporting the losses.
        
        # Plot our produced fakes at the end of each epoch. 
        plt.axis("off")                                                                                
        plt.title("Fake Images")                                                                       
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))

        plt.savefig("fake_images2.png")                                 
        plt.show()    
        
        # Save the model and the loss values every 50 epochs.
        if epoch % 5 == 4:
            torch.save(crit.state_dict(), f"models/ncritic2_{epoch+1}")
            torch.save(gen.state_dict(), f"models/nw_generator2_{epoch+1}")
        
            with open(f"losses/c_losses2_{epoch+1}", "wb") as fp:
                pickle.dump(c_losses, fp)

            with open(f"losses/w_g_losses2_{epoch+1}", "wb") as fp:
                pickle.dump(g_losses, fp)
                
    return c_losses, g_losses


convert = tvt.Compose([tvt.ToTensor(), tvt.Normalize([0.5], [0.5])])
data = torchvision.datasets.MNIST(root="./MNIST", transform=convert, download=True)

_, split2 = torch.utils.data.random_split(data, [30000, 30000])
    
if __name__ == "__main__":

    train_dataloader = torch.utils.data.DataLoader(split2, batch_size=256, num_workers=4, shuffle=True, drop_last=True)

    crit = Critic()
    gen = Generator()

    # gen.load_state_dict(torch.load("models/w_generator_18"))
    # crit.load_state_dict(torch.load("models/critic_18"))

    gen_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    crit_params = sum(p.numel() for p in crit.parameters() if p.requires_grad)

    print("Number of parameters in the generator (millions):", gen_params / 1e6)
    print("Number of parameters in the discriminator (millions):", crit_params / 1e6)

    epochs = 200

    c_losses, g_losses = train_wgan(crit, gen, train_dataloader, epochs, 1e-3)

