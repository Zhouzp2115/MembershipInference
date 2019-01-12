# coding:utf-8
from torch import optim
from torch.autograd import Variable

import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle
import random


class AttackModel(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(AttackModel, self).__init__()

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 120)
        self.fc3 = nn.Linear(120, 80)
        self.fc4 = nn.Linear(80, 10)
        self.fc5 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class AttackDataLoader(data.Dataset):
    def __init__(self, dict):
        self.dataset = dict['data']
        self.labels = dict['labels']

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.labels[index]

        data = torch.FloatTensor(data)
        return data, label

    def __len__(self):
        return len(self.labels)


def dataProcess(dict):
    # 900 out 4500 in
    random.seed(100)

    in_count = 0
    out_count = 0

    result = []
    for k in range(10):
        result.append({'labels': [], 'data': []})

    for i in range(10):
        for j in range(len(dict[i]['labels'])):
            itemcount = len(result[i]['labels'])
            if itemcount < 2:
                insertPos = 0
            else:
                insertPos = random.randint(0, itemcount - 1)

            if dict[i]['labels'][j] == 0:
                if out_count < 900:
                    out_count += 1
                    result[i]['labels'].insert(insertPos, 0)
                    result[i]['data'].insert(insertPos, dict[i]['data'][j])

            else:
                if in_count < 900:
                    in_count += 1
                    result[i]['labels'].insert(insertPos, 1)
                    result[i]['data'].insert(insertPos, dict[i]['data'][j])
        in_count = 0
        out_count = 0

    return result


if __name__ == '__main__':

    # load all attack model train data
    file = open('/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/AttackModel/trainset', 'rb')
    trainData = pickle.load(file, encoding='latin1')

    trainData = dataProcess(trainData)

    for count in range(10):
        attack = AttackModel()
        attack.train(mode=True)

        if torch.cuda.is_available():
            attack.cuda()

        print ('attack model %d trainset size = %d' % (count, len(trainData[count]['labels'])))

        dataLoader = AttackDataLoader(trainData[count])
        trainloader = torch.utils.data.DataLoader(dataLoader, batch_size=36, shuffle=True, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(attack.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(36):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = attack(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 50 == 49:  # 每100个batch打印一次训练状态
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0
        print('Finished Training AttackModel')

        torch.save(attack,
                   '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/AttackModel/' + 'attackmodel_' + str(
                       count) + '.pkl')
        print ('save attack model ok')
