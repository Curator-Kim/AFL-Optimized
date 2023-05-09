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
if not torch.cuda.is_available():
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')


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
# Define function hyperparameters
epochs=100
batch_size=32
lr=0.0004
input_dim=max_filesize
n_layer=2
hidden_dim=750
n_class=1

class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):                  
        out, _ = self.lstm(x)                                                            
        out = F.tanh(self.linear(out))             
        return out
 
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


def load_samples_bitmaps(sample_dir,bitmap_file,bitmap_dir='map_readelf'):
    Samples,max_filesize=Data_Processing.get_x(sample_dir,True)
    if os.path.exists(bitmap_file):
        bitmaps=eval(open(bitmap_file,'r').read())
    else:
        print('---------Getting bitmap data now... Wait for a moment...---------')
        bitmaps=Data_Processing.get_Bitmap_data_fast(bitmap_dir,bitmap_file,True)
    return Samples,np.array(bitmaps)

samples,bitmaps=load_samples_bitmaps(data_dir,bitmap_file)
print(bitmaps,len(bitmaps))
mean_bitmaps=bitmaps.sum()/len(bitmaps)
std_bitmaps=bitmaps.std()
# print(mean_bitmaps,std_bitmaps)

bitmaps=torch.tensor(bitmaps,dtype=torch.float).reshape(len(bitmaps),1,-1)
bitmap_normalizer = transforms.Normalize(mean=mean_bitmaps, std=std_bitmaps)
bitmaps=bitmap_normalizer(bitmaps)

# divide test and train set
train_size, test_size = int(len(samples) * 0.9), len(samples) - int(len(samples) * 0.9)  
train_sample_dataset, test_sample_dataset = random_split(samples, [train_size, test_size])
train_bitmap_dataset, test_bitmap_dataset = random_split(bitmaps, [train_size, test_size])
# train_sample_dataset = test_sample_dataset = samples
# train_bitmap_dataset = test_bitmap_dataset = bitmaps
print(train_size,test_size)

# Get dataloader
Train_loader = get_dataloader(batch_size,train_sample_dataset,train_bitmap_dataset)
Test_loader = get_dataloader(batch_size,test_sample_dataset,test_bitmap_dataset)

#
model=LSTMnet(in_dim=input_dim,hidden_dim=hidden_dim,n_layer=n_layer,n_class=n_class)
model=model.to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma = 0.8)


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
    # print('outputs: {}'.format(outputs*std_bitmaps+mean_bitmaps))
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