# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf 
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

def dataset(directory,size,batchsize):#定义函数，创建数据集
    """ parse  dataset."""
    def _parseone(example_proto):                         #解析一个图片文件
        """ Reading and handle  image"""
        #定义解析的字典
        dics = {}
        dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
        dics['img_raw'] = tf.FixedLenFeature(shape=[],dtype=tf.string)

        parsed_example = tf.parse_single_example(example_proto,dics)#解析一行样本
        
        image = tf.decode_raw(parsed_example['img_raw'],out_type=tf.uint8)
        image = tf.reshape(image, size)
        image = tf.cast(image,tf.float32)*(1./255)-0.5 #对图像数据做归一化
        
        label = parsed_example['label']
        label = tf.cast(label,tf.int32)
        label = tf.one_hot(label, depth=2, on_value=1) #转为0ne-hot编码
        return image,label
    
    dataset = tf.data.TFRecordDataset(directory)
    dataset = dataset.map(_parseone)
    dataset = dataset.batch(batchsize) #批次划分数据集
    
    dataset = dataset.prefetch(batchsize)
                
    return dataset


#如果显示有错，可以尝试使用np.reshape(thisimg, (size[0],size[1],3))或
#np.asarray(thisimg[0], dtype='uint8')改变类型与形状
def showresult(subplot,title,thisimg):          #显示单个图片
    p =plt.subplot(subplot)
    p.axis('off') 
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #显示 
    plt.figure(figsize=(20,10))     #定义显示图片的宽、高
    plt.axis('off')  
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])  
    plt.show() 

def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素  
    return one_element

sample_dir=['mydata.tfrecords']
size = [256,256,3]
batchsize = 10
tdataset = dataset(sample_dir,size,batchsize)

print(tdataset.output_types)  #打印数据集的输出信息
print(tdataset.output_shapes)

one_element1 = getone(tdataset)				#从tdataset里取出一个元素

with tf.Session() as sess:	# 建立会话（session）
    sess.run(tf.global_variables_initializer())  #初始化
    try:
        for step in np.arange(1):
            value = sess.run(one_element1)
            showimg(step,value[1],np.asarray( (value[0]+0.5)*255,np.uint8),10)       #显示图片        
    except tf.errors.OutOfRangeError:           #捕获异常
        print("Done!!!")
