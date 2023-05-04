import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import os
# helper deconv function
data_dir="Samples"
#get maximun of file size
maxsize=0
seed_names = os.listdir(data_dir)
for name in seed_names:
    tmp_size=os.path.getsize(data_dir+'/'+name)
    if tmp_size>maxsize:
        maxsize=tmp_size
maxsize+=32-maxsize%32

class Generator(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the Generator Module
        """
        super(Generator, self).__init__()
        self.hidden_size=hidden_size
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc=nn.Linear(hidden_size,output_size)
        self.tanh=nn.Tanh()


    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A maximun size Tensor as output
        """
        out,_=self.lstm(x)
        out=self.batch_norm(out.squeeze())
        out=self.fc(out.reshape(out.shape[0],1,-1))
        out=self.tanh(out)
        return out


d_conv_dim = 32
g_conv_dim = 32
z_size = 256


G=torch.load('Generator.pt')
device = "cuda:0" if torch. cuda.is_available() else "cpu"
#get random tensor
z = np.random.uniform(-1, 1, size=(16, 1, z_size))
z = torch.from_numpy(z).float()
# move x to GPU, if available
z = z.to(device)
G_Seed = G(z)

G_Seed=G_Seed.detach().cpu().numpy()
G_Seed=((G_Seed+1)*255/2).astype(np.uint8)
np.save('G_Seed',G_Seed)
print(G_Seed.shape)
print(G_Seed)
open('elf','wb').write(G_Seed[2][0])