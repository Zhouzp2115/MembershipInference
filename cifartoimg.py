import os
import numpy as np
from scipy.misc import imsave


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert(src, dstDir):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    imgData = unpickle(src)

    for i in range(len(imgData['labels'])):
        img = imgData['data'][i]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)

        img_name = dstDir + str(i) + '_label_' + str(imgData['labels'][i]) + '.jpg'
        imsave(img_name, img)


if __name__ == '__main__':
    root = '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/TargetModel/'

    src = root + 'trainset'
    dst = root + 'ImgTrain/'
    convert(src, dst)

    src = root + 'testset'
    dst = root + 'ImgTest/'
    convert(src, dst)

    root = '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/ShadowModelData/'
    for i in range(5):
        src = root + 'shadow_'+str(i)+'_train'
        dst = root + 'ImgTrain_'+str(i)+'/'
        convert(src, dst)

        src = root + 'shadow_' + str(i) + '_test'
        dst = root + 'ImgTest_' + str(i) + '/'
        convert(src, dst)