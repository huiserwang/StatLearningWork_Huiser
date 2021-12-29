import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlexNet(nn.Module):
    def __init__(self, width_mult=1) -> None:
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3, padding=1),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(512,20)

    def forward(self, x):
        # x: [32,100,15]
        N,HW,C = x.shape[:]
        x = x.permute(0,2,1).reshape(N, C, 10, 10)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, C*2*2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data, label) -> None:
        super(MyDataset, self).__init__()
        self.data = data
        self.label = label
        self.totensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.totensor(img)
        img = img.reshape(10,10,15).permute(2,0,1)
        label = self.label[idx]
        label = self.totensor(label)
        return img, label


    def __len__(self):
        return len(self.label)

class libCNN():
    def __init__(self, datas, labels, batchsize=32, epoch=10, lr=0.1) -> None:
        if datas.dims == 2:
            datas = datas.reshape(-1,100,15)
            labels = labels.reshape(-1, 100)[:,0]
        self.epoch=epoch
        self.batchsize=batchsize
        self.lr=lr
        self.model = AlexNet().to(device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        trainset = MyDataset(datas, labels)
        evalset = MyDataset(datas, labels)
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batchsize,
            shuffle=True
        )
        self.evalloader = torch.utils.data.DataLoader(
            evalset,
            batch_size=1,
            shuffle=False
        )


