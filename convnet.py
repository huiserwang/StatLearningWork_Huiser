import torch
import torch.nn as nn
# import torchvision.models as models
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def op_load_csv(path):
    f = open(path, 'r')
    content = f.readlines()
    content = [x.strip() for x in content]
    content = content[1:]
    data = [(x.split(',')[0], int(x.split(',')[1])) for x in content]
    return data

def op_load_npy(path):
    data = np.load(path)
    return data

def op_write_csv(test_files, test_pred_labels, out):
    all_out = []
    all_out.append('id,category')
    for i,each in enumerate(test_files):
        all_out.append('{},{}'.format(each[7:],int(test_pred_labels[i])))
    content = '\n'.join(all_out)
    f = open(out, 'w')
    f.writelines(content)
    f.close()

def op_merge_data(labels, prefix, concat=True):
    """merge data
    Args:
        labels (list(tuple)): labels for each file in csv
        prefix (str): where to load npy
        concat (bool, optional): if concat all data along dimension 0 into one array. Defaults to True.
    Returns:
        label_data (np.ndarray | list(np.ndarray)): labels for all frames, shape [1334*100,] if concat=True
        feats_data (np.ndarray | list(np.ndarray)): feats for all frames, shape [1334*100, 15] if concat=True
    """
    label_data = []
    feats_data = []
    for each in labels:
        npy_data = op_load_npy(os.path.join(prefix, each[0]))
        label_data.append(each[1])
        feats_data.append(npy_data)
    if concat:
        label_data = np.concatenate(label_data)
        feats_data = np.concatenate(feats_data)
    return np.stack(label_data), np.stack(feats_data).transpose(0,2,1)

def preprocess_data(label, feats, mode='norm'):
    """preprocess func
    Args:
        mode (str, optional): what you want to do. Defaults to 'norm'.
    Returns:
        label_data (np.ndarray): labels for all frames, shape [1334*100,]
        feats_data (np.ndarray): feats for all frames, shape [1334*100, 15]
    """
    if mode == 'norm':
        N = label.shape[0] if isinstance(label, np.ndarray) else len(label)*label[0].shape[0]
        label_data = label
        max_f = feats.max(axis=1)
        min_f = feats.min(axis=1)
        f_range = max_f - min_f
        feats_data = (feats - feats.min(axis=1).reshape(N,1)) / f_range.reshape(N,1)
        unkeep = np.unique(np.argwhere(np.isnan(feats_data)==True)[:,0])
        keep = np.ones(len(label_data)).astype(np.bool)
        keep[unkeep] = False
    elif mode == 'horizon_norm':
        pass
    else:
        print('error!')
        return None
    return label_data[keep], feats_data[keep], f_range, min_f

def process_test_data(feats, mode='norm'):
    if mode == 'norm':
        N = feats.shape[0]
        max_f = feats.max(axis=1)
        min_f = feats.min(axis=1)
        f_range = max_f - min_f
        feats_data = (feats - feats.min(axis=1).reshape(N,1)) / f_range.reshape(N,1)
        unkeep = np.unique(np.argwhere(np.isnan(feats_data)==True)[:,0])
        keep = np.ones(N).astype(np.bool)
        keep[unkeep] = False
    elif mode == 'horizon_norm':
        pass
    else:
        print('error!')
        return None
    return feats_data[keep]

def cal_label(labels):
    return np.argmax(np.bincount(np.array(labels).astype(np.int32)))

def to_cuda(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cuda()
    else:
        return x.cuda()

class dataset(object):
    def __init__(self,feats, labels=None) -> None:
        super().__init__()
        self.length = feats.shape[0]
        self.feats = feats
        self.labels = labels


    def __len__(self):
        return self.length

    def __getitem__(self,index):
        if self.labels is not None:
            return self.feats[index], self.labels[index]
        else:
            return self.feats[index]

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, (3,1), stride=(1,3)),
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, (3,1), stride=(1,3)),
                                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 1, 2, 1),
                                nn.ReLU())
        self.bottleneck1 = nn.Sequential(nn.Conv2d(32,32,3,1,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,1,padding=1),nn.ReLU())
        self.bottleneck2 = nn.Sequential(nn.Conv2d(32,64,3,1,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,1,padding=1),nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 20)

    def forward(self,x):
        x1 = self.conv1(x.unsqueeze(1))
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.bottleneck1(x3)
        x5 = self.bottleneck2(x3+x4)
        x6 = self.avgpool(x5)
        x7 = torch.flatten(x6, 1)
        out = self.fc(x7)
        return out

class libCNN():
    def __init__(self, train_dir, test_dir, label_train, epoch=500):
        self.label = op_load_csv(label_train)
        self.labels,self.feats = op_merge_data(self.label, train_dir, concat=False)
        index = []
        for i in range(20):
            # fix range to compare
            index.append(np.random.choice(np.argwhere(self.labels==i)[:,0],13,replace=False))
            # index.append(np.argwhere(labels==i)[:,0][:13])
        val_index = np.stack(index).reshape(-1)
        whole_index = np.arange(self.labels.shape[0])
        train_index = np.setdiff1d(whole_index,val_index,assume_unique=True)
        train_labels = self.labels[train_index]
        train_feats = self.feats[train_index]

        self.val_feats = self.feats[val_index]
        self.val_labels = self.labels[val_index]

        self.test_file_names = sorted(glob(os.path.join(test_dir,"*")))
        test_feats = []
        for i in self.test_file_names:
            test_origin_feats = np.load(i)
            test_feats.append(test_origin_feats)
        self.test_feats = np.stack(test_feats).transpose(0,2,1) # 348 100 15

        self.model = CNNModel().cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,200,300], gamma=0.9)
        self.start = 0 
        self.num_epochs = epoch

        train_dataset = dataset(train_feats, train_labels)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,num_workers=0,pin_memory=True,drop_last=True)

    def train(self,):
        self.best_score = 0.
        self.best_epoch = 0.
        if not os.path.isfile('ConvNet.ckpt'):
            for i in tqdm(range(self.start, self.num_epochs)):
                self.model.train()
                for feat, label in self.train_dataloader:
                    feat = to_cuda(feat).float()
                    label = to_cuda(label).long()
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    out = self.model(feat)
                    loss = self.criterion(out, label)
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()
                # writer.add_scalar('loss', loss.item(), global_step=i)
                if not (i+1)%100: 
                    print(loss.item())

                if not i%1:
                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(torch.from_numpy(self.val_feats).cuda().float())
                        preds = torch.argmax(out, dim=-1)
                        acc = (self.val_feats.shape[0] - torch.abs(torch.from_numpy(self.val_labels).cuda()-preds).bool().sum())/self.val_feats.shape[0]
                        print(acc)
                        if acc > self.best_score:
                            self.best_score = acc
                            self.best_epoch = i
                            torch.save(self.model.state_dict(),'ConvNet_best.ckpt')

            print('the best epoch is {}, score is {}'.format(self.best_epoch, self.best_score))
            

            print("eval mode!")
            st = torch.load('ConvNet_best.ckpt')
            self.model.load_state_dict(st)
            self.model.eval()
            with torch.no_grad():
                out = self.model(torch.from_numpy(self.feats).cuda().float())
                preds = torch.argmax(out, dim=-1)
                acc = (self.feats.shape[0] - torch.abs(torch.from_numpy(self.labels).cuda()-preds).bool().sum())/self.feats.shape[0]
                print(acc)
                # ------ eval test -------
                out = self.model(torch.from_numpy(self.test_feats).cuda().float())
                preds = torch.argmax(out, dim=-1).cpu().numpy().tolist()
                op_write_csv(self.test_file_names, preds, 'ConvNet_500_epoch_test_results.csv')
        
    def test(self):
        print("eval mode!")
        st = torch.load('ConvNet_best.ckpt')
        self.model.load_state_dict(st)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(self.feats).cuda().float())
            preds = torch.argmax(out, dim=-1)
            acc = (self.feats.shape[0] - torch.abs(torch.from_numpy(self.labels).cuda()-preds).bool().sum())/self.feats.shape[0]
            print(acc)
            # ------ eval test -------
            out = self.model(torch.from_numpy(self.test_feats).cuda().float())
            preds = torch.argmax(out, dim=-1).cpu().numpy().tolist()
            print(preds)
            op_write_csv(self.test_file_names, preds, 'ConvNet_500_epoch_test_results.csv')