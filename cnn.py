import torch
from torch._C import dtype
from torch.optim import optimizer
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
        conv1 = nn.Conv2d(15,32,kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(conv1.weight)
        #torch.nn.init.uniform_(conv1.weight, a=0, b=1)
        #torch.nn.init.uniform_(conv1.bias, a=0, b=1)
        self.layer1 = nn.Sequential(
            conv1,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(conv2.weight)
        self.layer2 = nn.Sequential(
            conv2,
            nn.ReLU(inplace=True)
        )

        conv3=nn.Conv2d(64,128,kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(conv3.weight)
        self.layer3 = nn.Sequential(
            conv3,
        )

        conv4 = nn.Conv2d(128,256,kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(conv4.weight)
        self.layer4 = nn.Sequential(
            conv4,
        )

        conv5 = nn.Conv2d(256,256,kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(conv5.weight)
        self.layer5 = nn.Sequential(
            conv5,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(256*2*2, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,20)
    
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # x: [32,15,10,10]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256*2*2)
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
        img = img.reshape(10,10,15).permute(2,0,1).to(dtype=torch.float32)
        label = self.label[idx]
        label = torch.tensor(label).to(dtype=torch.int64)
        return img, label

    def __len__(self):
        return len(self.label)

class MyDatasetTest(Dataset):
    def __init__(self, data, n=137) -> None:
        super(MyDatasetTest, self).__init__()
        self.data = data
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.num = n

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.totensor(img)
        img = img.reshape(10,10,15).permute(2,0,1).to(dtype=torch.float32)
        return img


    def __len__(self):
        return self.num

class libCNN():
    def __init__(self, datas, labels, datasTest, batchsize=32, epoch=10, lr=0.001) -> None:
        if len(datas.shape) == 2:
            datas = datas.reshape(-1,100,15)
            labels = labels.reshape(-1, 100)[:,0]
            datasTest = datasTest.reshape(-1, 100, 15)
        self.epoch=epoch
        self.batchsize=batchsize
        self.lr=lr
        self.model = AlexNet().to(device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        trainset = MyDataset(datas, labels)
        evalset = MyDataset(datas, labels)
        testset = MyDatasetTest(datasTest, n=348)
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
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False
        )

    def train(self):
        for epoch in range(self.epoch):
            sum_loss=0.0
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                outs = self.model(inputs)
                loss = self.loss(outs, labels)
                loss.backward()

                self.optimizer.step()

                sum_loss += loss.item()
                if i%10 == 9:
                    print("[{}, {}] loss: {}".format(epoch+1, i+1, sum_loss/10))
                    sum_loss = 0.0

    def eval(self):
        with torch.no_grad():
            correct = 0
            total = 0
            pred_labels = []
            for i, data in enumerate(self.evalloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outs = self.model(inputs)

                _, predicted = torch.max(outs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('{}'.format(float(correct/total)))

    def test(self):
        pred_label = []
        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                inputs = data
                inputs = inputs.to(device)

                outs = self.model(inputs)

                _, predicted = torch.max(outs.data, 1)
                pred_label.append(int(predicted))
        print('test done')
        return pred_label



