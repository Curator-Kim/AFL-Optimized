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
epochs=1000
batch_size=32
lr=0.0004
input_dim=max_filesize
n_layer=2
hidden_dim=1024
n_class=1
beta1=0.5
beta2=0.999

def scale(x, feature_range=(-1, 1)):
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min,max = feature_range
    x = x * (max-min) + min
    return x

class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim//2, n_layer, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):                  
        h0 = torch.zeros(self.n_layer*2, x.size(0), self.hidden_dim//2).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.n_layer*2, x.size(0), self.hidden_dim//2).to(device)
        out, _ = self.lstm(x, (h0,c0))                                                            
        out = F.tanh(self.linear(out[:, -1, :]))            
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
    train_size, test_size = int(len(samples) * 0.9), len(samples) - int(len(samples) * 0.9)  
    Train_set, Test_set = random_split(seed_data, [train_size, test_size])
    Train_loader = torch.utils.data.DataLoader(Train_set,
                                          batch_size,
                                          shuffle=True)
    Test_loader = torch.utils.data.DataLoader(Test_set,
                                          batch_size,
                                          shuffle=True)
    return Train_loader, Test_loader


def load_samples_bitmaps(sample_dir,bitmap_file,bitmap_dir='map_readelf',XOR=False,HAVOC=False):
    Samples,max_filesize=Data_Processing.get_x(sample_dir, XOR, HAVOC)
    if os.path.exists(bitmap_file):
        bitmaps=eval(open(bitmap_file,'r').read())
        if len(bitmaps)!=len(Samples):
            print("Data length error! Please check your bitmapfile")
            exit(-1)
    else:
        print('---------Getting bitmap data now... Wait for a moment...---------')
        bitmaps=Data_Processing.get_Bitmap_data_fast(bitmap_dir,bitmap_file,HAVOC)
    return Samples,np.array(bitmaps)

samples,bitmaps=load_samples_bitmaps(data_dir, bitmap_file, XOR = False, HAVOC= False)
print(samples[0])
print(bitmaps,len(bitmaps))
mean_bitmaps=bitmaps.sum()/len(bitmaps)
std_bitmaps=bitmaps.std()
max_bitmaps=bitmaps.max()
# print(mean_bitmaps,std_bitmaps)

bitmaps=torch.tensor(bitmaps,dtype=torch.float).reshape(len(bitmaps),1,-1)
# bitmap_normalizer = transforms.Normalize(mean=mean_bitmaps, std=std_bitmaps)
# bitmaps=bitmap_normalizer(bitmaps)
bitmaps/=max_bitmaps
bitmaps=scale(bitmaps)


# Get dataloader
Train_loader, Test_loader = get_dataloader(batch_size,samples, bitmaps)

#
model=LSTMnet(in_dim=input_dim,hidden_dim=hidden_dim,n_layer=n_layer,n_class=n_class)
model=model.to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2))
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma = 0.8)


train_losses=[]
test_losses=[]
# 训练模型
first_epoch=0
first_epoch=Data_Processing.load_checkpoint('drive/MyDrive/Mutate_Opt_Module',first_epoch,model,optimizer,True)
for epoch in range(first_epoch,epochs):
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
    # save breakpoint
    if epoch %10 == 0 and epoch:
        Data_Processing.save_checkpoint('drive/MyDrive/Mutate_Opt_Module',epoch,model,optimizer,True)
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