# MembershipInference
code for membershipinference against macgine learning
paper:https://ieeexplore.ieee.org/abstract/document/7958568

use pytorch(GPU version)
data:CIFAR-10

85%.py:nn model
CIFAR10.py:nn model

traintarget.py:train target model

targetmodel_test.py:get accuracy of the target model and generate data for attack model test

cifartoimg.py:convert cifar data(python version) to img data 

dataloader.py:dataloader for target/shadow model

selectdata_shadow.py:select data for shadow model train/test

trainshadow.py:trian shadow model on the data that from selectdata_shadow.py

shadowmodel_test.py:get accuracy of the trained shadow model and generate train data for attack model

trainattack.py:train attack model on the data that from shadowmodel_test.py

attackmodel_test.py:get accuracy of the attack model



