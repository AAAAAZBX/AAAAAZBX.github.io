---
title: "Pytorch模型训练套路总结"
id: "2025-12-06-03"
date: "2025-12-06"
description: "总结Pytorch模型训练的完整流程，包括数据集加载、模型定义、训练和测试等步骤"
tags: ["AI", "pytorch", "模型训练", "深度学习"]
---

# **模型训练**

# 1.完整的模型训练套路

## （1）首先导入数据集


```python
import torchvision.datasets

train_data = torchvision.datasets.CIFAR10("dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10("dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))
```

    Files already downloaded and verified
    Files already downloaded and verified
    训练集的长度为：50000
    测试集的长度为：10000


## （2）利用DataLoader加载数据集


```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
```

##   （3） 搭建神经网络

规范写法中，神经网络的架构都被存放于一个叫做model的Python中，此外，为了验证模型中的参数设置是否正确，使用如下的方法进行验证，看输出参数是否是10个


```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class ZBX(nn.Module):
    def __init__(self):
        super(ZBX, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    zbx = ZBX()
    input = torch.ones((64, 3, 32, 32))
    output = zbx(input)
    print(output.shape)
```

    torch.Size([64, 10])


将上面的模型放入到model.py文件中以后，便可以在train.py文件中使用如下命令调用此模型
~~~
from model import *
~~~

## （4）创建网络模型

在train文件中直接使用下面的命令即可调用model文件中的模型


```python
zbx = ZBX()
```

## （5）损失函数

这里使用交叉熵损失函数


```python
loss_fn = nn.CrossEntropyLoss()
```

## （6）优化器


```python
learning_rate = 1e-2
optimizer = torch.optim.SGD(zbx.parameters(), lr=learning_rate)
```

## （7）设置训练网络的一些参数


```python
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
```

## （8）训练

训练框架（不包括测试）：
~~~python
for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))
~~~

## （9）对模型进行测试

训练+测试的框架：
~~~python
for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))

    total_test_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zbx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
    print("整体测试集上的Loss:{}".format(total_test_loss))
~~~

## （10）添加tensorboard以观察训练进度

训练+测试+数据可视化框架：
~~~python
# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zbx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
    total_test_step = total_test_step + 1
    print("整体测试集上的Loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
~~~

## （11）保存模型

在每一轮结束后添加下面的语句即可
~~~python
torch.save(zbx, "zbx_{}.pth".format(i+1))
print("模型已保存")
~~~

## （12）关于正确率（分类问题的衡量指标）

比如，我们有两个输入，2*input,放到模型中可以得到一个输出，比如这是一个二分类问题

第一个输入得到的结果为$[0.1, 0,2]$,第二个输入得到的结果为$[0.3, 0.4]$

$[x1, x2]$中的$x1$表示对第一个类别预测的概率，$x2$表示对第二个类别预测的概率

这样便可以得出，模型在第一个输出上得到的结果为1，第二个也为1，因为在第一个类别上输出的概率最大（这里以0为起始点）

假设真实输入的target中第一个是0类别，第二个是1类别

如何从$[0.1, 0,2]$，$[0.3, 0.4]$这个形式转化为$[1],[1]$这个形式呢？

可以使用argmax函数preds=([1],[1]) input_target=([0],[1])

比较preds 和input_target，得到的结果为[false, true]

使用sum函数，即[false, true].sum()得到结果为1



```python
import torch

outputs = torch.tensor([[0.1, 0.2],
                      [0.3, 0.4]])

print(outputs.argmax(1))
```

    tensor([1, 1])



```python
preds = outputs.argmax(1)
targets = torch.tensor([0, 1])
print(preds == targets)
```

    tensor([False,  True])



```python
print((preds == targets).sum())
```

    tensor(1)


根据上面的代码可以在训练过程中添加整体正确率这一参数，来观察模型的预测准确度

~~~python
# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 测试步骤开始
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zbx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    total_test_step = total_test_step + 1
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(zbx, "zbx_{}.pth".format(i+1))
    print("模型已保存")

writer.close()
~~~

## （13）一些细节

在训练开始之前，有些人会添加一句
~~~python
zbx.train()
~~~
表示训练开始

在测试开始之前，也会添加一句
~~~python
zbx.eval()
~~~
表示测试状态开始，如果你的模型中含有Dropout层等一些特殊层的话，在测试的时候会关掉

# 总结

model.py

~~~python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class ZBX(nn.Module):
    def __init__(self):
        super(ZBX, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    zbx = ZBX()
    input = torch.ones((64, 3, 32, 32))
    output = zbx(input)
    print(output.shape)
~~~

train.py

~~~python
import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

#准备数据集
train_data = torchvision.datasets.CIFAR10("dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10("dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 使用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
zbx = ZBX()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(zbx.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zbx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    total_test_step = total_test_step + 1
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(zbx, "zbx_{}.pth".format(i+1))
    # torch.save(zbx.state_dict(),"zbx_{}.pth".format(i+1))
    print("模型已保存")

writer.close()
~~~

# 2.使用GPU进行训练

## 第一种方式（不常用）

含有CUDA方法的只有以下几个部分有

1.网络模型

2.数据（输入，标注）

3.损失函数

在使用cuda方法是只需要条加上

~~~python
xxx = xxx.cuda()
~~~
这条语句即可

使用下面的代码train.py+model.py合并，分别在gpu和cpu上面训练，可以比较训练所花费的时间

~~~python
import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import time

#准备数据集
train_data = torchvision.datasets.CIFAR10("dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10("dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 使用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class ZBX(nn.Module):
    def __init__(self):
        super(ZBX, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

zbx = ZBX()
if torch.cuda.is_available():
    zbx = zbx.cuda() #网络模型转移到CUDA上面去

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(zbx.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = zbx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    total_test_step = total_test_step + 1
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(zbx, "zbx_{}.pth".format(i+1))
    # torch.save(zbx.state_dict(),"zbx_{}.pth".format(i+1))
    print("模型已保存")

writer.close()
~~~

在cpu上进行训练，每100轮所用时间最快为4秒,谷歌colab的cpu达到了每100轮10秒，转移到谷歌colab实验台的gpu上，每100轮训练大概仅需一秒

## 第二种方式（较为常用）

~~~python
import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import time

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#准备数据集
train_data = torchvision.datasets.CIFAR10("dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10("dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 使用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class ZBX(nn.Module):
    def __init__(self):
        super(ZBX, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

zbx = ZBX()
zbx = zbx.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(zbx.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = zbx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    total_test_step = total_test_step + 1
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(zbx, "zbx_{}.pth".format(i+1))
    # torch.save(zbx.state_dict(),"zbx_{}.pth".format(i+1))
    print("模型已保存")

writer.close()
~~~

# 3.完整的模型训练套路

利用已经训练好的模型，然后给它提供输入

首先将图片转化为模型要求的输入格式


```python
import torchvision.transforms
from PIL import Image

image_path = "D:\\Micro_Climate_Summer_Task\\Pytorch_For_Deep_Learning\\imgs\\dog.jpg"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)
```

    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=488x331 at 0x21C2D241190>
    torch.Size([3, 32, 32])


然后加载网络模型


```python
class ZBX(nn.Module):
    def __init__(self):
        super(ZBX, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("zbx_1.pth")
print(model)
```

    ZBX(
      (model): Sequential(
        (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Flatten(start_dim=1, end_dim=-1)
        (7): Linear(in_features=1024, out_features=64, bias=True)
        (8): Linear(in_features=64, out_features=10, bias=True)
      )
    )


最后便能得到每个种类所预测的概率


```python
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
```

    tensor([[-3.1763, -1.0918,  1.0762,  1.2570,  2.6293,  1.1636,  3.1605,  2.2872,
             -4.5042, -1.1769]])



```python
print(output.argmax(1))
```

    tensor([6])


此模型只被训练了一轮，要查看训练30轮的情况，可以在colab上直接复制模型

~~~python
import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import time

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#准备数据集
train_data = torchvision.datasets.CIFAR10("dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10("dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 使用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class ZBX(nn.Module):
    def __init__(self):
        super(ZBX, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

zbx = ZBX()
zbx = zbx.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(zbx.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------------第{}轮训练开始-------------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = zbx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}.Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = zbx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    total_test_step = total_test_step + 1
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(zbx, "zbx_{}.pth".format(i+1))
    # torch.save(zbx.state_dict(),"zbx_{}.pth".format(i+1))
    print("模型已保存")

writer.close()
~~~

运行以后再将图片传入30轮训练以后得到的模型中。

test.py

~~~python
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

image_path = "D:\\Micro_Climate_Summer_Task\\Pytorch_For_Deep_Learning\\imgs\\plane.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class ZBX(nn.Module):
    def __init__(self):
        super(ZBX, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("zbx_29.pth",map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image,  (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
~~~

完结撒花！！！
