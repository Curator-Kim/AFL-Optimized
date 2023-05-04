import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torch.utils.data as data
import Data_Processing
from torchvision import transforms

import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

data_dir="Samples/Samples"
bitmap_file='bit_map_geq'

max_filesize=0
seed_names = os.listdir(data_dir)
for name in seed_names:
    tmp_size=os.path.getsize(data_dir+'/'+name)
    if tmp_size>max_filesize:
        max_filesize=tmp_size
if max_filesize%32:
    max_filesize+=32-max_filesize%32
print(max_filesize)
# Define function hyperparameters
epochs=100
batch_size=32
lr=0.000001
input_dim=max_filesize
n_layer=2
conv_dim=32
n_class=1

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

class CNNnet(nn.Module):
    def __init__(self, conv_dim): 
        """
        :param conv_dim: The depth of the first convolutional layer
        """
        super(CNNnet, self).__init__()

        self.conv_dim = conv_dim
        self.cv1 = conv(1, self.conv_dim, 4, batch_norm=False)
        self.cv2 = conv(self.conv_dim, self.conv_dim*2, 4, batch_norm=True)
        self.cv3 = conv(self.conv_dim*2, self.conv_dim*4, 4, batch_norm=True)
        self.cv4 = conv(self.conv_dim*4, self.conv_dim*8, 4, batch_norm=True)
        self.fc1 = nn.Linear(self.conv_dim*max_filesize//2,1)
        
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # print(x.shape)
        x = F.leaky_relu(self.cv1(x),0.2)
        # print(x.shape)
        x = F.leaky_relu(self.cv2(x),0.2)
        # print(x.shape)
        x = F.leaky_relu(self.cv3(x),0.2)
        # print(x.shape)
        x = F.leaky_relu(self.cv4(x),0.2)
        # print(x.shape)
        x = x.view(-1,self.conv_dim*max_filesize//2)
        # print(x.shape)
        x = self.fc1(x)

        return x
 
## Data loading...

class MyDataset(data.Dataset):
    def __init__(self, samples, bitmaps, transform=None):
        self.data=torch.tensor(samples,dtype=torch.float).reshape(len(samples),1,-1)
        self.labels=bitmaps
    
    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
def get_dataloader(batch_size, data, targets):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of seeds in a batch
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """

    seed_data = MyDataset(data,targets)
    
    data_loader = torch.utils.data.DataLoader(seed_data,
                                          batch_size,
                                          shuffle=True)
    return data_loader


def load_samples_bitmaps(sample_dir,bitmap_file,bitmap_dir='mapData/mapData'):
    Samples,max_filesize=Data_Processing.get_x(sample_dir)
    if os.path.exists(bitmap_file):
        bitmaps=eval(open(bitmap_file,'r').read())
    else:
        print('---------Getting bitmap data now... Wait for a moment...---------')
        bitmaps=Data_Processing.get_Bitmap_data_fast(bitmap_dir,bitmap_file)
    return Samples,np.array(bitmaps)

samples,bitmaps=load_samples_bitmaps(data_dir,bitmap_file)
mean_bitmaps=bitmaps.sum()/len(bitmaps)
std_bitmaps=bitmaps.std()

bitmaps=torch.tensor(bitmaps,dtype=torch.float).reshape(len(bitmaps),1,-1)
bitmap_normalizer = transforms.Normalize(mean=mean_bitmaps, std=std_bitmaps)
bitmaps=bitmap_normalizer(bitmaps)

# divide test and train
train_size, test_size = int(len(samples) * 0.9), len(samples) - int(len(samples) * 0.9)  
train_sample_dataset, test_sample_dataset = random_split(samples, [train_size, test_size])
train_bitmap_dataset, test_bitmap_dataset = random_split(bitmaps, [train_size, test_size])  
print(train_size,test_size)

# Get dataloader
Train_loader = get_dataloader(batch_size,train_sample_dataset,train_bitmap_dataset)
Test_loader = get_dataloader(batch_size,test_sample_dataset,test_bitmap_dataset)

#
model=CNNnet(conv_dim)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma = 0.8)


train_losses=[]
test_losses=[]
# 训练模型
for epoch in range(epochs):
    model.train()
    train_loss=0
    for batch,(batch_data, batch_targets) in enumerate(Train_loader):
        batch_data,batch_targets=batch_data.to(device),batch_targets.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data).reshape(-1,1,1)
        outputs = outputs.to(device)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        train_loss+=loss.item()
        optimizer.step()
    train_loss/= batch
    train_losses.append(train_loss)
    scheduler.step()
    print('outputs: {}, batch_targets: {}'.format(outputs, batch_targets))
    print('Epoch [{}/{}], Loss: {:.6f}, Lr: {:.4f}'.format(epoch+1, epochs, train_loss, optimizer.param_groups[0]['lr']))
    # test

    model.eval()  
    test_loss = 0 
    with torch.no_grad():  
        for batch,(X, y) in enumerate(Test_loader):  
            X, y = X.to(device), y.to(device)  
            pred = model(X).reshape(-1,1,1)
            test_loss += criterion(pred, y).item() 
    test_loss /= batch 
    test_losses.append(test_loss)
    print(f"Test Error:  Avg loss: {test_loss:>8f} \n")

torch.save(model,'Mut_Opt.pt')

# draw the picture
plt.plot(train_losses, label='train_losses', alpha=0.5)
plt.plot(test_losses, label='test_losses', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()