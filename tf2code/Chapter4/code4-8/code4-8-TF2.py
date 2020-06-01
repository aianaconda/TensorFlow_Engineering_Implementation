# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import tensorflow as tf
import numpy as np
tf.compat.v1.disable_v2_behavior()
#在内存中生成模拟数据
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X为-1到1之间连续的100个浮点数
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
    return train_X, train_Y   #以生成器的方式返回


train_data = GenerateData()  

#将内存数据转化成数据集
dataset = tf.data.Dataset.from_tensor_slices( train_data )		#元祖
dataset2 = tf.data.Dataset.from_tensor_slices( {                #字典
        "x":train_data[0],
        "y":train_data[1]
        } )		#

batchsize = 10  #定义批次样本个数
dataset3 = dataset.repeat().batch(batchsize) #批次划分数据集

dataset4 = dataset2.map(lambda data: (data['x'],tf.cast(data['y'],tf.int32)) )#自定义处理数据集元素
dataset5 = dataset.shuffle(100)#乱序数据集
    
def getone(dataset):
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素  
    return one_element
    
one_element1 = getone(dataset)				#从dataset里取出一个元素
one_element2 = getone(dataset2)				#从dataset2里取出一个元素
one_element3 = getone(dataset3)				#从dataset3里取出一个批次的元素
one_element4 = getone(dataset4)				#从dataset4里取出一个批次的元素
one_element5 = getone(dataset5)				#从dataset5里取出一个批次的元素


def showone(one_element,datasetname):
    print('{0:-^50}'.format(datasetname))
    for ii in range(5):
        datav = sess.run(one_element)#通过静态图注入的方式，传入数据
        print(datasetname,"-",ii,"| x,y:",datav)
        
def showbatch(onebatch_element,datasetname):
    print('{0:-^50}'.format(datasetname))
    for ii in range(5):
        datav = sess.run(onebatch_element)#通过静态图注入的方式，传入数据
        print(datasetname,"-",ii,"| x.shape:",np.shape(datav[0]),"| x[:3]:",datav[0][:3])
        print(datasetname,"-",ii,"| y.shape:",np.shape(datav[1]),"| y[:3]:",datav[1][:3])
        
with tf.compat.v1.Session() as sess:	# 建立会话（session）
    showone(one_element1,"dataset1")
    showone(one_element2,"dataset2")
    showbatch(one_element3,"dataset3")
    showone(one_element4,"dataset4")
    showone(one_element5,"dataset5")
    

    

