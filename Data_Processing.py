import os
import numpy as np
data_dir='Samples'
def delete(data_dir):
    seed_names = os.listdir(data_dir)
    avg_size=0
    rate=0.1
    for name in seed_names:
        tmp_size=os.path.getsize(data_dir+'/'+name)
        avg_size+=tmp_size
    avg_size/=len(seed_names)
    del_num=0
    for name in seed_names:#删除过小的文件
        tmp_size=os.path.getsize(data_dir+'/'+name)
        if tmp_size<avg_size*rate:
            os.remove(data_dir+'/'+name)
            del_num+=1
    print('finish:delete %d files'%(del_num))

def vectorize(fname):
    """ 读取文件并向量化 """
    v=list(open(fname,'rb').read())
    return np.array(v)

def mutate_pos(v1,v2):
    """
    用于获取变异位置：v1为原种子，v2为变异后文件，输入均需要向量化
    返回0-1构成的向量，1代表此位置发生变异
    要求v1和v2长度相等
    例如:v1=[1 2 3]  v2=[1 2 4]，返回 [0 0 1]
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
    seed_names=os.listdir(dirname)
    orig_file=open(dirname+'/'+seed_names[0],'rb').read()
    for i in range(1,len(dirname)):
        cur_file=open(dirname+'/'+seed_names[i],'rb').read()
        x.append(mutate_pos(orig_file,cur_file))
    return x

def test():
    v1=vectorize('small_exec.elf')
    v2=vectorize('small_exec2.elf')
    print(v1,v2)
    print(mutate_pos(v1,v2))
    print(get_cov('Samples'))
    pass

print(get_x('Samples'))