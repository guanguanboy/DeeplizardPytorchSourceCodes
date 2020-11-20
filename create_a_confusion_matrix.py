import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
import os

torch.set_printoptions(linewidth=120) # Display options for output
torch.set_grad_enabled(True) # Already on by default

train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]))

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        #implement the forward pass,前向方法是输入张量到一个预测的输出张量的映射
        # 任何神经网络的输入层都是由输入数据决定的，例如：如果输入张量包含三个元素，那么网络将有三个节点包含在它的输入层中
        # 出于这个原因我们可以把输入层看作是恒等变换f(x) = x

        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t) # 有权重
        t = F.relu(t) #无权重
        t = F.max_pool2d(t, kernel_size=2, stride=2) #无权重

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #在从卷积层转换到全连接层时，有一个flatten张量的操作。
        # （4）hidden linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t

network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

#跟踪loss及正确分类的样本数目

"""
total_loss = 0
total_correct = 0

batch = next(iter(train_loader)) # Get Batch

for batch in train_loader:

    images, labels = batch

    preds = network(images) #Pass Batch
    loss = F.cross_entropy(preds, labels) # calculate the loss

    optimizer.zero_grad() # 告诉优化器把梯度属性中的梯度归0，这么做的原因是pytorch会累加梯度，所以在我们计算梯度之前，需要确保我们没有任何当前存在的梯度值
    loss.backward() #calculate gradients
    optimizer.step() # update weights

    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)

print("epoch:", 0, "total_correct:", total_correct, "loss:", total_loss)

print(total_correct / len(train_set))
"""

#trainning with multiple epochs: The complete trainning loop
for epoch in range(5):
    total_loss = 0
    total_correct = 0

    batch = next(iter(train_loader)) # Get Batch

    for batch in train_loader:

        images, labels = batch

        preds = network(images) #Pass Batch
        loss = F.cross_entropy(preds, labels) # calculate the loss

        optimizer.zero_grad() # 告诉优化器把梯度属性中的梯度归0，这么做的原因是pytorch会累加梯度，所以在我们计算梯度之前，需要确保我们没有任何当前存在的梯度值
        loss.backward() #calculate gradients
        optimizer.step() # update weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)

print(total_correct / len(train_set))




print(len(train_set))
print(len(train_set.targets))

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds

with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

#Using The Predictions Tensor
preds_correct = get_num_correct(train_preds, train_set.targets)
print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))

#build the confusion matrix
print(train_set.targets)
print(train_preds.argmax(dim=1))

stacked = torch.stack( (train_set.targets, train_preds.argmax(dim=1)), dim=1)

print(stacked.shape)

print(stacked)

print(stacked[0].tolist())

"""
Now, we can iterate over these pairs and count the number of occurrences at each position in the matrix. 
Let's create the matrix. 
Since we have ten prediction categories, we'll have a ten by ten matrix.
"""

cmt = torch.zeros(10, 10, dtype=torch.int64)
print(cmt)

for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1

print(cmt)

import matplotlib.pyplot as plt
from plotcm import plot_confusion_matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cmt.numpy(), train_set.classes)
plt.show()

"""
也可以直接使用sklean中的函数来计算cmt
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))

计算出来的cm直接是ndarray类型的
print(type(cm))

然后可以直接plot
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, train_set.classes)

参考自：https://deeplizard.com/learn/video/0LhiS6yu2qQ
"""