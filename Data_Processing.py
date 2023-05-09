import os
import numpy as np
import time
import torch
import time
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
data_dir='Samples'
mapdata_dir='mapData'

def load_checkpoint(path,default_epoch,modules,optimizers,verbose):
    """
    Try to load a checkpoint to resume the training.
    :param path:
        Path for your checkpoint file
    :param default_epoch:
        Initial value for "epoch" (in case there are not snapshots)
    :param modules:
        nn.Module containing the model or a list of nn.Module objects. They are assumed to stay on the same device
    :param optimizers:
        Optimizer or list of optimizers
    :param verbose:
        Verbose mode
    :return:
        Next epoch
    """
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(optimizers, optim.Optimizer):
        optimizers = [optimizers]

    # If there's a checkpoint
    if os.path.exists(path):
        # Load data
        data = torch.load(path, map_location=next(modules[0].parameters()).device)

        # Inform the user that we are loading the checkpoint
        if verbose:
            print(f"Loaded checkpoint saved at {datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S')}. "
                  f"Resuming from epoch {data['epoch']}")

        # Load state for all the modules
        for i, m in enumerate(modules):
            modules[i].load_state_dict(data['modules'][i])

        # Load state for all the optimizers
        for i, o in enumerate(optimizers):
            optimizers[i].load_state_dict(data['optimizers'][i])

        # Next epoch
        return data['epoch'] + 1
    else:
        return default_epoch
    
def save_checkpoint(path,
                    epoch,
                    modules,
                    optimizers,
                    safe_replacement):
    """
    Save a checkpoint of the current state of the training, so it can be resumed.
    This checkpointing function assumes that there are no learning rate schedulers or gradient scalers for automatic
    mixed precision.
    :param path:
        Path for your checkpoint file
    :param epoch:
        Current (completed) epoch
    :param modules:
        nn.Module containing the model or a list of nn.Module objects
    :param optimizers:
        Optimizer or list of optimizers
    :param safe_replacement:
        Keep old checkpoint until the new one has been completed
    :return:
    """

    # This function can be called both as
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, my_module, my_opt)
    # or
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, [my_module1, my_module2], [my_opt1, my_opt2])
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(optimizers, optim.Optimizer):
        optimizers = [optimizers]
 
    # Data dictionary to be saved
    data = {
        'epoch': epoch,
        # Current time (UNIX timestamp)
        'time': time.time(),
        # State dict for all the modules
        'modules': [m.state_dict() for m in modules],
        # State dict for all the optimizers
        'optimizers': [o.state_dict() for o in optimizers]
    }

    # Safe replacement of old checkpoint
    temp_file = None
    if os.path.exists(path) and safe_replacement:
        # There's an old checkpoint. Rename it!
        temp_file = path + '.old'
        os.rename(path, temp_file)

    # Save the new checkpoint
    with open(path, 'wb') as fp:
        torch.save(data, fp)
        # Flush and sync the FS
        fp.flush()
        os.fsync(fp.fileno())

    # Remove the old checkpoint
    if temp_file is not None:
        os.unlink(path + '.old')


def delete(data_dir, rate=0.1):
    """ delte files while are too small """
    seed_names = os.listdir(data_dir)
    avg_size=0
    for name in seed_names:
        tmp_size=os.path.getsize(data_dir+'/'+name)
        avg_size+=tmp_size
    avg_size/=len(seed_names)
    del_num=0
    for name in seed_names:
        tmp_size=os.path.getsize(data_dir+'/'+name)
        if tmp_size<avg_size*rate:
            os.remove(data_dir+'/'+name)
            del_num+=1
    print('finish:delete %d files'%(del_num))

def vectorize_from_file(fname):
    """ get the file and vectorize """
    v=list(open(fname,'rb').read())
    return np.array(v)


def padding(f,size):
    for _ in range(size-len(f)):
        f.append(0)
    return f

def get_maxsize(dirname):
    maxsize=0
    seed_names = os.listdir(dirname)
    for name in seed_names:
        tmp_size=os.path.getsize(dirname+'/'+name)
        if tmp_size>maxsize:
            maxsize=tmp_size
    return maxsize

def mutate_pos(v1,v2):
    """
    get the mutate position, len(v1) should be equal to len(v2)
    :param v1: vector1
    :param v2: vector2
    :return z 0-1 vector
    expample: v1=[1 2 3]  v2=[1 2 4]  z= [0 0 1]
    """
    if len(v1)!=len(v2):
        print("length of v1 should be equal to v2")
        exit(0)
    z=np.zeros(len(v1))
    for i in range(len(v1)):
        if v1[i]!=v2[i]:
            z[i]=1
    return z


def get_cov(dirname):
    seed_names=os.listdir(dirname)
    y=np.zeros(len(seed_names))
    for i in range(len(seed_names)):
        if(seed_names[i].split(',')[-1] == '+cov'):
            y[i]=1
    return y


def get_x(dirname, havoc=False):
    x=[]
    if os.path.exists(dirname)==False:
        print('File not exists!')
        exit(-1)
    seed_names=os.listdir(dirname)
    max_size=get_maxsize(dirname)
    if max_size%32:
        max_size+=32-max_size%32
    orig_file=list(open(dirname+'/'+seed_names[0],'rb').read())
    orig_file=np.array(padding(orig_file,max_size))
    for i in range(len(seed_names)):
        if havoc and 'havoc' in seed_names[i]:
            continue
        cur_file=list(open(dirname+'/'+seed_names[i],'rb').read())
        padding(cur_file,max_size)
        x.append(mutate_pos(orig_file,np.array(cur_file)))
    return x,max_size

# def get_Bitmap_data(dirname):
#     bitmap_data=[]
#     seed_names=os.listdir(dirname)
#     orig_bitmap=open(dirname+'/'+seed_names[0],'rb').read()
#     for i in range(50):
#         cur_bitmap=open(dirname+'/'+seed_names[i],'rb').read()
#         cur=0
#         for i in range(0x10000):
#             for j in range(8):
#                 if (cur_bitmap[i]>>j)&1 == 1 and (orig_bitmap[i]>>j)&1 == 0:
#                     cur+=1
#         bitmap_data.append(cur)
#     return bitmap_data

def get_Bitmap_data_fast(dirname, saved_file, havoc): #仅快一点点
    """ 
    a<b:
    (a^b)&a
    """
    bitmap_data=[]
    seed_names=os.listdir(dirname)
    orig_bitmap=open(dirname+'/'+seed_names[0],'rb').read()
    for i in range(len(seed_names)):
        if havoc and 'havoc' in seed_names[i]:
            continue
        cur_bitmap=open(dirname+'/'+seed_names[i],'rb').read()
        cur=0
        for j in range(0x10000):
            tmp = (cur_bitmap[j]^orig_bitmap[j])&cur_bitmap[j]
            cur += bin(tmp).count('1')
        bitmap_data.append(cur)
    open(saved_file,'w').write(str(bitmap_data))
    print('--------Finish getting bitmap data---------')
    return bitmap_data

def func2(dirname):
    filename=os.listdir(dirname)
    num=0
    for i in range(len(filename)):
        if "+cov" in filename[i]:
            num+=1
    return num,num/i

def func1(dirname):
    filename=os.listdir(dirname)
    num=0
    for i in range(len(filename)):
        f=open(dirname+'/'+filename[i],'rb').read()
        if f[0]==0x7f and f[1]==0x45 and f[2]==0x4c and f[3]==0x46:
            num+=1
    return num,num/i

def bit_num(dirname):
    table=[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 
4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]
    bitnum_data=[]
    seed_names=os.listdir(dirname)
    for i in range(len(seed_names)):
        cur_bitmap=open(dirname+'/'+seed_names[i],'rb').read()
        cur=0
        for j in range(0x10000):
            cur+=table[cur_bitmap[j]]
        bitnum_data.append(cur)
    return bitnum_data



def test():
    # v1=vectorize_from_file('small_exec.elf')
    # v2=vectorize_from_file('small_exec2.elf')
    # print(v1,v2)
    # print(mutate_pos(v1,v2))
    # print(get_cov('Samples'))
    # print(get_x('Samples'))
    # time1=time.time()
    # bit_map_geq=get_Bitmap_data_fast(mapdata_dir,'bit_map_geq')
    # time2=time.time()
    # print(time2-time1)
    # print(get_Bitmap_data(mapdata_dir))
    # time3=time.time()
    # print(time3-time2)
    # print(func2('Samples/Samples'))
    # print(func1("Samples/Samples"))
    bitmapdata=bit_num('mapData/mapData')
    print(bitmapdata)
    open('bitmapdata','w').write(str(bitmapdata))
    # Sample_filter('Samples/Samples','Filtered_Samples')
    pass

if __name__=='__main__':
    test()