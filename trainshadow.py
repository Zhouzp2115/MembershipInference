# coding:utf-8
from dataloader import CIFARDataLoader

from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积-> 激活 -> 池化
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    for count in range(3):
        print ('train shadow model ', count)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        CIFARDataset = CIFARDataLoader(
            '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/ShadowModelData/shadow_' + str(
                count) + '_train', transform)
        trainloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=150, shuffle=True, num_workers=2)

        net = Net()
        net.train(mode=True)

        if torch.cuda.is_available():
            net.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(150):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # 每100个batch打印一次训练状态
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
        print('Finished Training', count)

        torch.save(net,
                   '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/ShadowModelData/shadow_model_' + str(
                       count) + '.pkl')
        print ('save model ..... OK')
