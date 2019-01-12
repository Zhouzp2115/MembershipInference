# coding:utf-8
from dataloader import CIFARDataLoader
from traintarget import Net

from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import random
import pickle
import os

def saveData(dir, filename, dict):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir + filename):
        os.system(r'touch {}'.format(dir + filename))

    file = open(dir + filename, 'wb')
    pickle.dump(dict, file)
    print ('save data..........ok')


if __name__ == '__main__':

    random.seed(1200)

    trainsetOutput = []
    testsetOutput = []

    for i in range(10):
        trainsetOutput.append({'labels': [], 'data': []})
        testsetOutput.append({'labels': [], 'data': []})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # in test set
    CIFARDataset = CIFARDataLoader(
        '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/testset',
        3000, transform)
    testloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=1, shuffle=True, num_workers=2)

    target = torch.load(
        '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/targetmodel.pkl').cuda()
    target.eval()

    total = 0
    correct = 0

    for i, data in enumerate(testloader, 0):
        images, labels = data
        index = labels

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        outputs = target(images)

        # get data for attack model test
        testsetOutput[index]['labels'].append(0)
        testsetOutput[index]['data'].append(outputs.cpu().data.numpy()[0])

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print ('in testset:correct=%d total=%d acc=%f' % (correct, total, correct / total))

    # in train set
    CIFARDataset = CIFARDataLoader(
        '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/trainset',
        15000, transform)
    trainloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=1, shuffle=True, num_workers=2)
    total = 0
    correct = 0

    for i, data in enumerate(trainloader, 0):
        images, labels = data
        index = labels

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        outputs = target(images)

        # get data for attack model test
        trainsetOutput[index]['labels'].append(1)
        trainsetOutput[index]['data'].append(outputs.cpu().data.numpy()[0])

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print ('in trainset:correct=%d total=%d acc=%f' % (correct, total, correct / total))

    for i in range(10):
        print ('label_%d testset outputsize = %d   trainset outputsize = %d' % (i, len(testsetOutput[i]['labels']),
                                                                                len(trainsetOutput[i]['labels'])))

    saveData('/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/', 'trainset_output',
             trainsetOutput)
    saveData('/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/', 'testset_output',
             testsetOutput)
