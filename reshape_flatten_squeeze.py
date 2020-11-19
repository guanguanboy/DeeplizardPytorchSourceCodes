import torchvision
import torch

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])

t.size()

t.shape

len(t.shape)

torch.tensor(t.shape).prod()

t.numel() # number of elements in tensor

t.reshape(4,3) #通过reshape操作不改变张量的维数，rank数目，轴的个数
print(len(t.shape))

t.reshape(6,2)
print(len(t.shape))
t.reshape(12, 1)
print(len(t.shape))

#另一种改变张量的方法是通过squeeze 和 unsqueeze
# squeeze 一个张量可以移除所有长度为1的轴，
# unsqueeze 一个张量可以则会增加一个长度为1的张量
# 注意squeeze 和 unsqueeze 可以改变张量的维数，rank数目，轴的个数。允许我们扩大或者缩小张量的秩
print(t.reshape(1,12))
print(t.reshape(1,12).shape)

print(t.reshape(1, 12).squeeze())
print(t.reshape(1, 12).squeeze().shape)
print(len(t.reshape(1, 12).squeeze().shape)) # 这里张量的rank数目变成了1，改变了张量的维数，轴的个数

print(t.reshape(1, 12).squeeze().unsqueeze(dim=0))
print(t.reshape(1,12).squeeze().unsqueeze(dim=0).shape)
print(len(t.reshape(1,12).squeeze().unsqueeze(dim=0).shape)) # unsqueeze又恢复了张量的rank

print('unsqueeze on dim = 1')
print(t.reshape(1, 12).squeeze().unsqueeze(dim=1)) #在列的位置上增加一维
print(t.reshape(1,12).squeeze().unsqueeze(dim=1).shape)
print(len(t.reshape(1,12).squeeze().unsqueeze(dim=1).shape))


#flatten a tensor,意味着除去所有的轴，只保留一个轴，这个操作创造了一个单轴的张量，这个新的张量
#包含了原来张量中所有的元素。相当于我们创建了一个一维数组，这个数组包含了张量的所有标量分量
#一个flatten函数可以这样定义
def flatten(t):
    t = t.reshape(1, -1) # -1表示，根据一个张量中包含的元素个数来求出-1这个位置上的实际的值应该是多少
    t = t.squeeze()
    return t

print(flatten(t)) # flatten()相当于reshape and squeeze

print(t.reshape(1, 12)) # only reshape

#torch cat
t1 = torch.tensor([
    [1,2],
    [3,4]
])

t2 = torch.tensor([
    [5,6],
    [7,8]
])

# combine t1 and t2 row-wise(axis-0) in the following way, row-wise,改变行的数目，不改变列的数目
print(torch.cat((t1, t2), dim=0))

# combine t1 and t2 column-wise(axis-1) like this: colun-wise,改变列的数目，不改变行的数目。
print(torch.cat((t1, t2), dim=1))