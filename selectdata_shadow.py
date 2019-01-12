# coding:utf-8

import pickle
import os


def unPickle(fileDir):
    fo = open(fileDir, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    return dict


def saveData(dir, filename, dict):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir + filename):
        os.system(r'touch {}'.format(dir + filename))

    file = open(dir + filename, 'wb')
    pickle.dump(dict, file)
    print ('save file..........ok')


if __name__ == '__main__':
    fileDir = '/home/zhouzp/MachineLearning/CIFAR10/cifar-10-batches-py/SortedData/'

    trainData = []
    testData = []

    for i in range(10):
        trainFile = unPickle(fileDir + 'label_' + str(i) + '_train')
        testFile = unPickle(fileDir + 'label_' + str(i) + '_test')

        trainFile['labels'] = trainFile['labels'][1500:5000]
        trainFile['data'] = trainFile['data'][1500:5000]
        trainFile['filenames'] = trainFile['filenames'][1500:5000]
        testFile['labels'] = testFile['labels'][300:1000]
        testFile['data'] = testFile['data'][300:1000]
        testFile['filenames'] = testFile['filenames'][300:1000]

        trainData.append(trainFile)
        testData.append(testFile)

    # shadow model个数:3 每个训练集大小:15000
    # 训练集每类取法 0-1500 1000-2500 2000-3500
    # 测试集每类取法 0-300  200-500   400-700

    sdmodelTrainData = []
    sdmodelTestData = []

    for i in range(3):
        sdmodelTrainData.append({'batch_label': 'training batch 1 of 5', 'labels': [], 'data': [], 'filenames': []})
        sdmodelTestData.append({'batch_label': 'training batch 1 of 5', 'labels': [], 'data': [], 'filenames': []})

    for k in range(3):
        for i in range(1500):
            for j in range(10):
                sdmodelTrainData[k]['labels'].append(trainData[j]['labels'][k * 1000 + i])
                sdmodelTrainData[k]['data'].append(trainData[j]['data'][k * 1000 + i])
                sdmodelTrainData[k]['filenames'].append(trainData[j]['filenames'][k * 1000 + i])
        for i in range(300):
            for j in range(10):
                sdmodelTestData[k]['labels'].append(testData[j]['labels'][k * 200 + i])
                sdmodelTestData[k]['data'].append(testData[j]['data'][k * 200 + i])
                sdmodelTestData[k]['filenames'].append(testData[j]['filenames'][k * 200 + i])


    for i in range(3):
        saveData(fileDir + 'ShadowModelData/', 'shadow_' + str(i) + '_train', sdmodelTrainData[i])
        saveData(fileDir + 'ShadowModelData/', 'shadow_' + str(i) + '_test', sdmodelTestData[i])
