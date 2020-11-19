import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
import os

torch.set_printoptions(linewidth=120) # Display options for output
torch.set_grad_enabled(True) # Already on by default

print(torch.__version__)
print(torchvision.__version__)

"""
## The Trainning Process
1. get batch from the training set.
2. Pass batch to network.
3. Calculate the loss(difference between the predicted values and the true values). 使用loss function 完成
4. Calculate the gradient fo the loss function w.r.t the network's weight. 使用反向传播完成
5. Update the weights using the gradients to reduce the loss. 使用优化算法完成
6. Repeat steps 1-5 until one epoch is completed. 通过循环完成
7. Repeat steps 1-6 for as many epochs required to obtain the desired level of accuracy.  通过循环完成

"""
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

train_set = torchvision.datasets.FashionMNIST(root='/mnt/liguanlin/DataSets/Datasets/FashionMNIST',
                                              train=True,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor()
                                              ]))


# trainning with all batches: A single epoch
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
