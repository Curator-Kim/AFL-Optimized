import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

import os
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import torch
import torch.optim as optim
import torch.utils.data as data
device = "cuda:0" if torch. cuda.is_available() else "cpu"
data_dir="Samples"
#get maximun of file size
maxsize=0
seed_names = os.listdir(data_dir)
for name in seed_names:
    tmp_size=os.path.getsize(data_dir+'/'+name)
    if tmp_size>maxsize:
        maxsize=tmp_size


# Define hyperparameters
batch_size = 32
z_size = 100
input_size=z_size
hidden_size=128

d_conv_dim = 32
g_conv_dim = 32
maxsize+=g_conv_dim-maxsize%g_conv_dim
output_size=maxsize
print(maxsize)
lr = 0.0005
beta1 = 0.3
beta2 = 0.999

class SeedDataset(data.Dataset):
    def __init__(self, seed_dir, transform=None):
        self.seed_dir = seed_dir
        self.transform = transform
        self.seed_names = os.listdir(seed_dir)
    
    def __getitem__(self, index):
        seed_name = self.seed_names[index]
        seed_path = os.path.join(self.seed_dir, seed_name)
        seed = list(open(seed_path,'rb').read())
        seed=self.padding(seed)
        seed = torch.tensor(seed,dtype=torch.float)
        seed=seed.reshape(1,-1)
        if self.transform:
            seed = self.transform(seed)
        return seed
    
    def __len__(self):
        return len(self.seed_names)
    
    def padding(self, l):
        for _ in range(maxsize-len(l)):
            l.append(0)
        return l
    
def get_dataloader(batch_size, data_dir):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of seeds in a batch
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """

    seed_data = SeedDataset(data_dir)
    
    data_loader = torch.utils.data.DataLoader(seed_data,
                                          batch_size,
                                          shuffle=True)
    return data_loader



#get a dataloader
train_loader = get_dataloader(batch_size, data_dir)
# obtain one batch of training seeds
dataiter = iter(train_loader)
seeds= dataiter.next() # _ for no labels

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))

def scale(x, feature_range=(-1, 1)):
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min,max = feature_range
    x = x * (max-min) + min
    return x

# check scaled range
# should be close to -1 to 1

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv1d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm1d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, conv_dim): #conv_dim=32
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim
        #32 x 32
        self.cv1 = conv(1, self.conv_dim, 4, batch_norm=False)
        #16 x 16
        self.cv2 = conv(self.conv_dim, self.conv_dim*2, 4, batch_norm=True)
        #4 x 4
        self.cv3 = conv(self.conv_dim*2, self.conv_dim*4, 4, batch_norm=True)
        #2 x 2
        self.cv4 = conv(self.conv_dim*4, self.conv_dim*8, 4, batch_norm=True)
        
        self.fc1 = nn.Linear(self.conv_dim*8*2*2,1)
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        x = F.leaky_relu(self.cv1(x),0.2)
        x = F.leaky_relu(self.cv2(x),0.2)
        x = F.leaky_relu(self.cv3(x),0.2)
        x = F.leaky_relu(self.cv4(x),0.2)
        
        x = x.view(-1,self.conv_dim*8*2*2)
        x = self.fc1(x)
        return x

class Generator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Generator Module
        """
        super(Generator, self).__init__()
        self.hidden_size=hidden_size
        self.lstm=nn.LSTM(input_size,hidden_size)
        self.fc=nn.Linear(hidden_size,output_size)
        self.tanh=nn.Tanh()


    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A maximun size Tensor as output
        """
        out,_=self.lstm(x)
        out=self.fc(out[-1])
        out=self.tanh(out)
        return out

# def weights_init_normal(m):
#     """
#     Applies initial weights to certain layers in a model .
#     The weights are taken from a normal distribution 
#     with mean = 0, std dev = 0.02.
#     :param m: A module or layer in a network    
#     """
#     classname = m.__class__.__name__
#     if 'Linear' in classname:
#         torch.nn.init.normal_(m.weight,0.0,0.02)
#         m.bias.data.fill_(0.01)
#     # Apply initial weights to convolutional and linear layers
#     if 'Conv' in classname or 'BatchNorm1d' in classname:
#         torch.nn.init.normal_(m.weight,0.0,0.02)


def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(input_size=z_size,hidden_size=g_conv_dim,output_size=output_size)

    # initialize model weights
    # D.apply(weights_init_normal)
    # G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G

D, G = build_network(d_conv_dim, g_conv_dim, z_size)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')

def real_loss(D_out,smooth=False):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


# Create optimizers for the discriminator D and generator G
d_optimizer = optim.Adam(D.parameters(), lr, betas=(beta1, beta2))
g_optimizer = optim.Adam(G.parameters(), lr, betas=(beta1, beta2))

def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size, 32))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, real_images in enumerate(train_loader):

            batch_size = real_images.size(0)

            real_images = scale(real_images)
            
            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()

            # real images
            real_images = real_images.to(device)#[32,1,20480]
            

            dreal = D(real_images)
            dreal_loss = real_loss(dreal)

            #fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size, 32))
            
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            z = z.to(device)
            fake_images = G(z)
            print(fake_images.shape)

            # loss of fake images           
            dfake = D(fake_images)
            dfake_loss = fake_loss(dfake)
            
            #Adding both lossess
            d_loss = dreal_loss + dfake_loss
            #Backpropogation step
            d_loss.backward()
            d_optimizer.step() 

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            z = z.to(device)
            fake_images = G(z)

            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake, True) # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step() 

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        # generate and save sample, fake images
        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    # finally return losses
    return losses

# set number of epochs 
n_epochs = 20


# call training function
losses = train(D, G, n_epochs=n_epochs)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()

torch.save(G,'Generator.pt')