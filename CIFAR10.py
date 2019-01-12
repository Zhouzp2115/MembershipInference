# coding: utf-8
 
# In[1]:
 
 
#模块准备
from torch.autograd import Variable
 
import torch as t
 
import torchvision as tv 
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() #把Tensor变成Image，方便可视化
 
 
# In[2]:
 
 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
 
 
# In[3]:
 
 
#训练集
trainset = tv.datasets.CIFAR10(
                    root='/home/zhouzp/MachineLearning/CIFAR10',
                    train=True,
                    download=False,
                    transform=transform
)
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=500,
    shuffle=True,
    num_workers=2
)
#测试集
testset = tv.datasets.CIFAR10(
                    root='/home/zhouzp/MachineLearning/CIFAR10',
                    train=False,
                    download=False,
                    transform=transform
)
testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)
 
 
# In[4]:
 
 
classes = ('plane', 'car', 'bird', 'cat', 
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
 
# In[5]:
 
 
(data, label) = trainset[500]
print(classes[label])
 
#(data + 1) / 2是为了还原被归一化的数据，程序输出的图片如下图所示
 
#class torchvision.transforms.Normalize(mean, std)
#给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。即：Normalized_image=(image-mean)/std。
show((data + 1) / 2).resize((100,100))
 
 
# In[6]:
 
 
dataiter = iter(trainloader) #iter是迭代构造器，参数是object。iter返回的是一个iterator类型，有next()函数
images, labels = dataiter.next() #返回4张图片及标签，如下图所示。next()函数指的是，每执行一次，都会指向下一组数据
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400,100))
 
 
# In[7]:
 
 
import torch.nn as nn
import torch.nn.functional as F
 
class Net(nn.Module):
    def __init__(self):
        #nn.Module子类的函数必须在构造函数中执行父类的构造函数
        #下式等价于nn.Module.__init__(self)
        super(Net,self).__init__()
        #super是比较两个类谁更高，谁是父类，然后执行父类的函数
        #卷积层1表示输入图片为单通道，6表示输出通道数，5表示卷积核为5*5
        self.conv1 = nn.Conv2d(3,6,5)
        #卷积层
        self.conv2 = nn.Conv2d(6,16,5)
        #仿射层/全连接层，y = Wx + b
        #考虑到池化，所以线性全连接层的像素数减半
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        #问题来了：x是什么类型？是tensor还是variable
        #卷积-> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #按道理讲，（2,2）池化和2池化是一样的
        #reshape,'-1'表示自适应,自适应得到的应为16×25的矩阵
        #在这里有一个问题，tensor的维数是从小到大排列吗？0对应最小的维数索引，等等等
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()
if t.cuda.is_available():
    net.cuda()  
print(net)
 
 
# In[8]:
 
 
#定义损失函数和优化器
from torch import optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
 
 
# In[9]:
 
 
#训练网络：包括输入数据，前向传播加反向传播，更新参数
for epoch in range(100):
    
    running_loss = 0.0    
    for i, data in enumerate(trainloader, 0): #enumerate是枚举的意思
        
#         #输入数据
#         inputs, labels = data
#         inputs, labels = Variable(inputs), Variable(labels)
        
#         #梯度清零
#         optimizer.zero_grad()
        
        #前向传播和反向传播
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
      
            #更新参数
#         optimizer.step()
        
        #下边是cuda的用法
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        if t.cuda.is_available(): #在这也出过错，if后边不能少了冒号
            inputs = inputs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad() #最后在这出错了，这是个函数，得有括号
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
            
    
        #打印log信息
        running_loss += loss.item()
        if i % 100 == 99: #每2000个batch打印一次训练状态
            print('[%d, %5d] loss: %.3f'                  % (epoch+1, i+1, running_loss / 100))
            #格式化输出
            running_loss = 0.0
print('Finished Training')
 
 
# In[10]:
 
 
dataiter = iter(testloader)
images, labels = dataiter.next()
print('实际的label：', ' '.join(                           '%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100))
 
 
# In[11]:
 
 
#计算图片上每个类别的分数
images = Variable(images)
labels = Variable(labels)
if t.cuda.is_available():
    images = images.cuda()
    labels = labels.cuda()
outputs = net(images)
#得分最高的那个类
_, predicted = t.max(outputs.data, 1) 
#max(data, n)指的是在n维上求最大值。因为outputs.data的形状是4×10，所以，是在10维上求最大值。
 
print('预测结果：', ' '.join('%5s'                       % classes[predicted[j]] for j in range(4)))
 
 
# In[12]:
 
 
correct = 0 #预测正确的图片数
total = 0 #总共的图片数
for data in testloader:
    images, labels = data
    images = Variable(images)
 
    if t.cuda.is_available():
        images = images.cuda()
    outputs = net(images)
    _,predicted = t.max(outputs.data, 1) #第一次在这出错了，predicted写成了pridicted
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum() #第二次在这出错了，labels并没有传到GPU上，所以没有办法计算
    
print('1000张测试集中的准确率维：%d %%' % (100 * correct / total))
 
 
# In[13]:
 
 
correct = 0 #预测正确的图片数
total = 0 #总共的图片数
for data in trainloader:
    images, labels = data
    images = Variable(images)
    if t.cuda.is_available:
        images = images.cuda()
    outputs = net(images)
    _,predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('1000张测试集中的准确率维：%d %%' % (100 * correct / total))
