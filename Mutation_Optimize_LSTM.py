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

data_dir="Samples"
bitmap_file='bit_map_geq'

max_filesize=0
seed_names = os.listdir(data_dir)
for name in seed_names:
    tmp_size=os.path.getsize(data_dir+'/'+name)
    if tmp_size>max_filesize:
        max_filesize=tmp_size

# Define function hyperparameters
epochs=8000
batch_size=32
lr=0.001
input_dim=max_filesize
n_layer=3
hidden_dim=550
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
        out = out[:, -1, :]                
                                           
        out = self.linear(out)             
        return out
 
## Data loading...

class MyDataset(data.Dataset):
    def __init__(self, samples, bitmaps, transform=None):
        self.data=torch.tensor(samples,dtype=torch.float).reshape(len(samples),1,-1)
        self.labels=torch.tensor(bitmaps,dtype=torch.float).reshape(len(bitmaps),1,-1)
        labels_mean=torch.mean(self.labels)
        labels_std=torch.std(self.labels)
        self.labels=(self.labels-labels_mean)/labels_std
    
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


def load_samples_bitmaps(sample_dir,bitmap_file):
    Samples,max_filesize=Data_Processing.get_x(sample_dir)
    if os.path.exists(bitmap_file):
        bitmaps=eval(open(bitmap_file,'r').read())
    else:
        bitmaps=Data_Processing.get_Bitmap_data_fast(bitmap_file)
    return Samples,np.array(bitmaps)

samples,bitmaps=load_samples_bitmaps(data_dir,bitmap_file)
mean_bitmaps=bitmaps.sum()/len(bitmaps)
std_bitmaps=bitmaps.std()
bitmap_normalizer = transforms.Normalize(mean=mean_bitmaps, std=std_bitmaps)

train_size, test_size = int(len(samples) * 0.8), len(samples) - int(len(samples) * 0.8)  
train_sample_dataset, test_sample_dataset = random_split(samples, [train_size, test_size])
train_bitmap_dataset, test_bitmap_dataset = random_split(bitmaps, [train_size, test_size])  
print(train_size,test_size)

Train_loader = get_dataloader(batch_size,train_sample_dataset,train_bitmap_dataset)
Test_loader = get_dataloader(batch_size,test_sample_dataset,test_bitmap_dataset)
# ##
model=LSTMnet(in_dim=input_dim,hidden_dim=hidden_dim,n_layer=n_layer,n_class=n_class)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses=[]
test_losses=[]
# 训练模型
for epoch in range(epochs):
    model.train()
    for batch_data, batch_targets in Train_loader:
        batch_data,batch_targets=batch_data.to(device),batch_targets.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data).reshape(-1,1,1)
        outputs = outputs.to(device)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    print('outputs: {}'.format(outputs*std_bitmaps+mean_bitmaps))
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
    # test
    size = len(Test_loader.dataset)  
    num_batches = len(Test_loader)  

    model.eval()  
    test_loss = 0 
    with torch.no_grad():  
        for X, y in Test_loader:  
            X, y = X.to(device), y.to(device)  
            pred = model(X).reshape(-1,1,1)
            test_loss += criterion(pred, y.long()).item()  
    test_loss /= num_batches  
    test_losses.append(test_loss)
    print(f"Test Error:  Avg loss: {test_loss:>8f} \n")

torch.save(model,'Mut_Opt')

# draw the picture
plt.plot(train_losses, label='train_losses', alpha=0.5)
plt.plot(test_losses, label='test_losses', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()