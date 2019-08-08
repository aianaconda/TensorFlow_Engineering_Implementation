
# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle

Dataset =tf.data.Dataset

def load_sample(sample_dir,shuffleflag = True):
    '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
    print ('loading sample  dataset..',sample_dir)
    if not os.path.exists(sample_dir):
        print("there is not a dir path:",sample_dir)
        fullpath = os.getcwd()
        sample_dir = os.sep.join([fullpath, sample_dir]) 
        print(sample_dir)
        if not os.path.exists(sample_dir):
            raise IOError("there is not a dir path:"+sample_dir)
        
    lfilenames = []
    labelsnames = []
    fullpath = os.getcwd()
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#递归遍历文件夹 
        #print(dirpath,dirnames,filenames)        
        for filename in filenames:                            #遍历所有文件名
            filename_path =fullpath+'/'+dirpath+'/'+filename#+ os.sep.join([dirpath, filename]) 
            #print(filename_path)
            lfilenames.append(filename_path)               #添加文件名
            labelsnames.append( dirpath.split('/')[-1] )#添加文件名对应的标签
            
    lab= list(sorted(set(labelsnames)))  #生成标签名称列表
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #生成字典
    print('标签对应的字典：',labdict)

    labels = [labdict[i] for i in labelsnames]#no 0 yes 1
    #print("要返回的结果:",lfilenames,labels,lab)
    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)

def mydataset(directory,batch=32,shuffleflag=True,mode = 'train'):# 路径 尺寸
    """load and parse  dataset."""
    
    (filenames,labels),labelsnames = load_sample(directory,shuffleflag=False)   #载入文件名称与标签

    print("总共：",len(labelsnames),filenames[0],labels)


#    filenames = filenames[np.where(labels==0)]#只取恶意的
#    labels= labels[np.where(labels==0)]#只取恶意的
#    print(filenames)

    
    #转成字符串+标签
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    def _parseone(filename, label):                         #解析一个图片文件
        """ Reading and handle  image"""
        datasettxtlab = tf.data.TextLineDataset(filename).map(lambda x: (x,label  ))
        return datasettxtlab
    dataset = dataset.flat_map( _parseone )# map里面返回dataset时需要用flat_map
    
    #去掉无效字符
    def myfilter(x,y):
        def checkone(line):
            if len(line) <=2:
                return False
            for ch in line:
                #print(line,ch)
                if ch<32 or ch>=127:
                    return False
            return True
        isokstr = tf.py_func( checkone, [x], tf.bool)
        return isokstr
    dataset = dataset.filter(myfilter)		  #过滤掉非ascII字符

    if mode=='train':
        if shuffleflag == True:                                                   #对数据进行乱序操作
            dataset = dataset.shuffle(buffer_size=1000000)  
            dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(1)

        

    iterator = dataset.make_one_shot_iterator()#创建生成器。取一条
    print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
    print(dataset.output_shapes)
    next_element = iterator.get_next()
    return next_element


