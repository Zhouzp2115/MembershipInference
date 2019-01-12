# coding:utf-8
from dataloader import CIFARDataLoader
from trainshadow import Net
from torch import optim
from torch.autograd import Variable

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import random
import os
import pickle


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

    attackTrainSet = []
    for i in range(10):
        attackTrainSet.append({'labels': [], 'data': []})

    for count in range(3):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # in test set
        CIFARDataset = CIFARDataLoader(
            '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/ShadowModelData/shadow_' + str(
                count) + '_test', transform)
        testloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=1, shuffle=True, num_workers=2)

        target = torch.load(
            '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/ShadowModelData/shadow_model_' + str(
                count) + '.pkl').cuda()
        target.eval()

        total = 0
        correct = 0

        for i, data in enumerate(testloader, 0):
            images, labels = data
            index = labels

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            outputs = target(images)

            # get attack train set
            # print (outputs.cpu().data.numpy()[0])
            itemCount = len(attackTrainSet[index]['labels'])
            if itemCount == 0 or itemCount == 1:
                insertPos = 0
            else:
                insertPos = random.randint(0, len(attackTrainSet[index]['labels']) - 1)
            attackTrainSet[index]['labels'].insert(insertPos, 0)
            attackTrainSet[index]['data'].insert(insertPos, outputs.cpu().data.numpy()[0])

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        print ('shadow_model_%d in testset:correct=%d total=%d acc=%f' % (count, correct, total, correct / total))

        # in train set
        CIFARDataset = CIFARDataLoader(
            '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/ShadowModelData/shadow_' + str(
                count) + '_train', transform)
        trainloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=1, shuffle=True, num_workers=2)
        total = 0
        correct = 0

        for i, data in enumerate(trainloader, 0):
            images, labels = data
            index = labels

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            outputs = target(images)

            # get attack train data
            itemCount = len(attackTrainSet[index]['labels'])
            if itemCount == 0 or itemCount == 1:
                insertPos = 0
            else:
                insertPos = random.randint(0, len(attackTrainSet[index]['labels']) - 1)

            attackTrainSet[index]['labels'].insert(insertPos, 1)
            attackTrainSet[index]['data'].insert(insertPos, outputs.cpu().data.numpy()[0])

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print ('shadow_model_%d in trainset:correct=%d total=%d acc=%f' % (count, correct, total, correct / total))

    for i in range(10):
        print ('attack_model_%d trainset: labels-%d items data-%d items' % (
            i, len(attackTrainSet[i]['labels']), len(attackTrainSet[i]['data'])))

    saveData('/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/AttackModel/', 'trainset',
             attackTrainSet)
