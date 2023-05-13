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
import Data_Processing
# Check for a GPU
device = "cuda:0" if torch. cuda.is_available() else "cpu"

if not torch.cuda.is_available():
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')

data_dir="Samples/Samples"
#get maximun of file size
maxsize=0
seed_names = os.listdir(data_dir)
for name in seed_names:
    tmp_size=os.path.getsize(data_dir+'/'+name)
    if tmp_size>maxsize:
        maxsize=tmp_size
if maxsize%32:
    maxsize+=32-maxsize%32
print(maxsize)
# Define hyperparameters
batch_size = 64
z_size = 160
input_size = z_size
G_hidden_size = 1024
D_hidden_size = maxsize
G_layers = 3
D_layers = 3
n_epochs = 15000
# epoch_soft_label=n_epochs
clip_value=0.01
n_critic=5
output_size = maxsize

d_lr = 0.0001
g_lr = 0.0001
beta1 = 0.5
beta2 = 0.999


# if epoch_soft_label >= n_epochs:
#     print("Not using soft labels")
# else:
#     print("After%d epochs, using soft labels"%epoch_soft_label)

class SeedDataset(data.Dataset):
    def __init__(self, seed_dir, transform=None):
        self.seed_dir = seed_dir
        self.transform = transform
        self.seed_names = os.listdir(seed_dir)
        self.seeds=self.get_seeds(seed_dir)
    
    def __getitem__(self, index):
        return self.seeds[index]
    
    def __len__(self):
        return len(self.seed_names)
    
    def padding(self, l):
        for _ in range(maxsize-len(l)):
            l.append(0)
        return l
    
    def get_seeds(self,seed_dir):
        seeds=[]
        seed_names = os.listdir(seed_dir)
        for i in range(len(seed_names)):
            cur_file=list(open(seed_dir+'/'+seed_names[i],'rb').read())
            cur_file=self.padding(cur_file)
            cur_file=torch.tensor(cur_file,dtype=torch.float)
            cur_file=cur_file/255
            cur_file=cur_file.reshape(1,-1)
            if self.transform:
                cur_file=self.transform(cur_file)
            seeds.append(cur_file)
        return seeds
    
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

def scale(x, feature_range=(-1, 1)):
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min,max = feature_range
    x = x * (max-min) + min
    return x

def get_activation_fn(fn):
    activation_fn_list = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(),
        "prelu": nn.PReLU(),
        "rrelu": nn.RReLU(),
        "elu": nn.ELU(),
        "softplus": nn.Softplus()
    }
    return activation_fn_list[fn]

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers): #num_layers not used yet
        """
        Initialize the Discriminator Module
        """
        super(Discriminator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                get_activation_fn("leakyrelu")
            ),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                get_activation_fn("leakyrelu")
            ),
            nn.Sequential(
                nn.Linear(hidden_size, 1),
            )
        ])
        
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        for layer in self.layers:
            res = x
            x = layer(x)
            if res.shape[-1] == x.shape[-1]:
                x = x + res
        return x

class Generator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """
        Initialize the Generator Module
        """
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                get_activation_fn("relu")
            ),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                get_activation_fn("relu")
            ),
            nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Tanh()
            )
        ])


    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A maximun size Tensor as output
        """
        for layer in self.layers:
            res = x
            x = layer(x)
            if res.shape[-1] == x.shape[-1]:
                x = x + res
        return x
    
def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    classname = m.__class__.__name__
    if 'Linear' in classname:
        torch.nn.init.normal_(m.weight,0.0,0.02)
        m.bias.data.fill_(0.01)
    if 'Conv1d' in classname or 'BatchNorm1d' in classname:
        torch.nn.init.normal_(m.weight,0.0,0.02)

def build_network(G_hidden_size, D_hidden_size, G_layers, D_layers, z_size, pretrain=False, pretrain_G='Generator_pre.pt'):
    # define discriminator and generator
    D = Discriminator(output_size, D_hidden_size, D_layers)
    D.apply(weights_init_normal)
    if pretrain:
        if os.path.exists(pretrain_G):
            print('Used pretrained generator')
            G=torch.load(pretrain_G)
        else:
            print("The pretrained generator not exists!")
    else:
        G = Generator(input_size=z_size, hidden_size=G_hidden_size, output_size=output_size, num_layers=G_layers)
        G.apply(weights_init_normal)
    print(D)
    print()
    print(G)
    return D, G

pretrain=False
D, G = build_network(G_hidden_size, D_hidden_size, G_layers, D_layers, z_size, pretrain=pretrain)

# Create optimizers for the discriminator D and generator G
d_optimizer = optim.RMSprop(D.parameters(), d_lr) #不要用基于动量的优化算法（包括momentum和Adam）
g_optimizer = optim.RMSprop(G.parameters(), g_lr)

#  Dynamically adjusting learning rate -- Fixed step attenuation
d_scheduler = optim.lr_scheduler.StepLR(d_optimizer,step_size=500,gamma = 0.9)
g_scheduler = optim.lr_scheduler.StepLR(g_optimizer,step_size=500,gamma = 0.9)

def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    
    D=D.to(device)
    G=G.to(device)
    first_epoch=0
    first_epoch=Data_Processing.load_checkpoint('drive/MyDrive/Seed_Opt_Module',first_epoch,[G,D],[g_optimizer,d_optimizer],True)
   
    losses = [] # keep track of loss

    # Get some fixed data for sampling. These are seeds that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, 1, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    fixed_z = fixed_z.to(device)

    # epoch training loop
    for epoch in range(first_epoch, n_epochs):

        # batch training loop
        for batch_i, real_seeds in enumerate(train_loader):
            
            batch_size = real_seeds.size(0)
            random_noise = np.random.uniform(0, 0.003, size=(batch_size, 1, maxsize))
            random_noise = torch.from_numpy(random_noise).float()
            real_seeds = real_seeds + random_noise
            # print(real_seeds)
            real_seeds = scale(real_seeds)
            # 1. Train the discriminator on real and fake seeds
            d_optimizer.zero_grad()

            real_seeds = real_seeds.to(device)

            # Generate fake seeds
            z = np.random.uniform(-1, 1, size=(real_seeds.shape[0], 1, z_size))
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            z = z.to(device)
            fake_seeds = G(z).detach() #少计算一次 generator 的所有参数的梯度，同时，也不必刻意保存一次计算图，占用不必要的内存。discriminator 比 generator 简单才使用此方案
            # print(fake_seeds)   

            d_loss = -torch.mean(D(real_seeds)) + torch.mean(D(fake_seeds))#改进2、生成器和判别器的loss不取log
            #Backpropogation step
            d_loss.backward()
            d_optimizer.step() 
            # Clip weights of discriminator
            for p in D.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator with an adversarial loss
            g_optimizer.zero_grad()
            
            # Generate fake seeds
            z = np.random.uniform(-1, 1, size=(real_seeds.shape[0], 1, z_size))
            z = torch.from_numpy(z).float()
            z = z.to(device)
            fake_seeds = G(z)

            # Compute the discriminator losses on fake seeds 
            # using flipped labels!
            D_fake = D(fake_seeds)
            g_loss = -torch.mean(D_fake)

            # perform backprop
            g_loss.backward()
            g_optimizer.step() 
            

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f} | g_lr: {:6.6f} | d_lr: {:6.6f}'.format(
                        epoch+1, n_epochs, d_loss.item(),  g_loss.item(), 
                        g_optimizer.state_dict()['param_groups'][0]['lr'], d_optimizer.state_dict()['param_groups'][0]['lr']))


        ## AFTER EACH EPOCH##    
        # generate and save sample, fake seeds

        #     # Adjust learning rate
        g_scheduler.step()
        d_scheduler.step()
        G.eval() # for generating samples
        samples_z = G(fixed_z).detach().cpu().numpy()
        samples_z=((samples_z+1)/2*255).astype(np.uint8)
        print(samples_z)
        G.train() # back to training mode
        if epoch %10==0 and epoch:
            Data_Processing.save_checkpoint('drive/MyDrive/Seed_Opt_Module',epoch,[G,D],[g_optimizer,d_optimizer],True)

    # finally return losses
    return losses


# call training function
losses = train(D, G, n_epochs=n_epochs)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()
torch.save(G,'drive/MyDrive/Generator_WGAN.pt')