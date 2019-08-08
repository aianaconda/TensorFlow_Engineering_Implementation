# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""



import tensorflow as tf
Dataset =tf.data.Dataset

def parse_fn(x):
    print(x)
    return x

dataset = (Dataset.list_files('testset\*.txt', shuffle=False)
               .interleave(lambda x:
                   tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
                   cycle_length=2, block_length=2))




def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素
    return one_element

one_element1 = getone(dataset)				#从dataset里取出一个元素


def showone(one_element,datasetname):
    print('{0:-^50}'.format(datasetname))
    for ii in range(20):
        datav = sess.run(one_element)#通过静态图注入的方式，传入数据
        print(datav)



with tf.Session() as sess:	# 建立会话（session）
    showone(one_element1,"dataset1")
