# coding:utf-8

from trainattack import AttackModel
from trainattack import AttackDataLoader

from torch import optim
from torch.autograd import Variable

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import random
import pickle
import os

if __name__ == '__main__':
    # load test data
    file = open('/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/trainset_output', 'rb')
    testData_in = pickle.load(file, encoding='latin1')
    file = open('/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/testset_output', 'rb')
    testData_out = pickle.load(file, encoding='latin1')

    # test membership inference attack acc
    for count in range(10):
        # acc at trainset_output
        attack = torch.load(
            '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/AttackModel/attackmodel_' + str(
                count) + '.pkl').cuda()
        attack.eval()

        correct_train = 0
        total_train = 0

        print ('attack model testData_in label = %d size = %d' % (count, len(testData_in[count]['labels'])))

        testLoader = AttackDataLoader(testData_in[count])
        trainloader = torch.utils.data.DataLoader(testLoader, batch_size=1, shuffle=True, num_workers=2)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            outputs = attack(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        print (
        'in target trainset:correct=%d total=%d acc=%f' % (correct_train, total_train, correct_train / total_train))

        # acc at testset_output
        correct_test = 0
        total_test = 0
        testLoader = AttackDataLoader(testData_out[count])
        trainloader = torch.utils.data.DataLoader(testLoader, batch_size=1, shuffle=True, num_workers=2)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            outputs = attack(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
        print ('in target testset:correct=%d total=%d acc=%f' % (correct_test, total_test, correct_test / total_test))

        correct_all = correct_test + correct_train
        total_all = total_test + total_train
        print ('all correct=%d total=%d acc=%f ' % (correct_all, total_all, correct_all / total_all))
        print ('')
