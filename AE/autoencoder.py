from cv2 import bitwise_xor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import hiddenlayer as hl
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, tanh
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

# 数据准备
train_data = MNIST(
    root='./data/MNIST',
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

# 将图像转化为向量
train_data_x = train_data.data.type(torch.FloatTensor) / 255.0
train_data_x = train_data_x.reshape(train_data_x.shape[0], -1)
train_data_y = train_data.targets

# 定义数据加载器
train_loader = Data.DataLoader(
    dataset=train_data_x,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

# 测试数据导入
test_data = MNIST(
    root='./data/MNIST',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = test_data_x.reshape(test_data_x.shape[0], -1)
test_data_y = test_data.targets
print('训练数据集：', train_data_x.shape)
print('测试数据集：', test_data_x.shape)


class EnDecoder(nn.Module):
    def __init__(self):
        super(EnDecoder, self).__init__()

        # 定义 encoder
        self.Encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.Tanh(),
        )

        # 定义 decoder
        self.Decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    # 定义网络的前向传播
    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder


edmodel = EnDecoder()
# print(edmodel)

# 定义优化器
optimizer = torch.optim.Adam(edmodel.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# history1 = hl.History()
# canvas1 = hl.Canvas()

train_num = 0
val_num = 0

for epoch in range(50):
    print('epoch:', epoch)
    train_loss_epoch = 0

    for step, b_x in enumerate(train_loader):
        _, output = edmodel(b_x)
        loss = loss_func(output, b_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * b_x.size(0)

        train_num = train_num + b_x.size(0)

    train_loss = train_loss_epoch / train_num
    print('loss==========', train_loss)
    # history1.log(epoch, train_loss=train_loss)

    # with canvas1:
    #     canvas1.draw_plot(history1['train_loss'])
