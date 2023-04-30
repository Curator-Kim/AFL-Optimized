import os
import numpy as np
import time
data_dir='Samples'
mapdata_dir='mapData'
def delete(data_dir):
    """ delte files while are too small """
    seed_names = os.listdir(data_dir)
    avg_size=0
    rate=0.1
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


def get_x(dirname):
    x=[]
    if os.path.exists(dirname)==False:
        print('File not exists!')
        exit(-1)
    seed_names=os.listdir(dirname)
    max_size=get_maxsize(dirname)
    orig_file=list(open(dirname+'/'+seed_names[0],'rb').read())
    orig_file=np.array(padding(orig_file,max_size))
    for i in range(len(seed_names)):
        cur_file=list(open(dirname+'/'+seed_names[i],'rb').read())
        padding(cur_file,max_size)
        x.append(mutate_pos(orig_file,np.array(cur_file)))
    return x,max_size

def get_Bitmap_data(dirname):
    bitmap_data=[]
    seed_names=os.listdir(dirname)
    orig_bitmap=open(dirname+'/'+seed_names[0],'rb').read()
    for i in range(50):
        cur_bitmap=open(dirname+'/'+seed_names[i],'rb').read()
        cur=0
        for i in range(0x10000):
            for j in range(8):
                if (cur_bitmap[i]>>j)&1 == 1 and (orig_bitmap[i]>>j)&1 == 0:
                    cur+=1
        bitmap_data.append(cur)
    return bitmap_data

def get_Bitmap_data_fast(dirname):
    bitmap_data=[]
    seed_names=os.listdir(dirname)
    orig_bitmap=open(dirname+'/'+seed_names[0],'rb').read()
    for i in range(len(seed_names)):
        cur_bitmap=open(dirname+'/'+seed_names[i],'rb').read()
        cur=0
        for j in range(0x10000):
            tmp = (cur_bitmap[j]^orig_bitmap[j])&cur_bitmap[j]
            cur += bin(tmp).count('1')
        bitmap_data.append(cur)
    return bitmap_data
""" 
a<b:
(a^b)&a
"""
def test():
    # v1=vectorize_from_file('small_exec.elf')
    # v2=vectorize_from_file('small_exec2.elf')
    # print(v1,v2)
    # print(mutate_pos(v1,v2))
    # print(get_cov('Samples'))
    # print(get_x('Samples'))
    time1=time.time()
    bit_map_geq=get_Bitmap_data_fast(mapdata_dir)
    time2=time.time()
    print(time2-time1)
    open('bit_map_geq','w').write(str(bit_map_geq))
    # print(get_Bitmap_data(mapdata_dir))
    # time3=time.time()
    # print(time3-time2)
    pass

if __name__=='__main__':
    test()