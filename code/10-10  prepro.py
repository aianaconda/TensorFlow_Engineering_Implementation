# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import os
import scipy.io
import scipy.misc
from collections import Counter
import pandas as pd
from pandas import Series,DataFrame
from sklearn.utils import shuffle


class preprosample(object):

    def __init__(self, sample_dir='./data'):

        self.sample_dir = sample_dir


    def load_txt_evaldata(self,evaldata):    
        for (dirpath, dirnames, filenames) in os.walk(evaldata):#一级一级的文件夹递归
            #print(dirpath,dirnames,filenames)
            sdata = []
            for filename in filenames:
                filename_path = os.sep.join([dirpath, filename]) 
                with open(filename_path, 'rb') as f:  
                    for onedata in f: 
                        onedata = onedata.strip(b'\n') 
                        try:
                            #print(onedata.decode('gb2312'))#,onedata.decode('gb2312'))'UTF-8'
                            sdata.append(onedata.decode( 'gb2312' ).replace('\r',''))
                        except (UnicodeDecodeError):
                            print("wrong:",onedata.decode)

        return np.array( sdata  )
                  

        
    def make_dictionary(self):

        words_dic = [chr(i) for i in range(32,127)]
        
        words_dic.insert(0,'None')#补0用的
        print(words_dic)
        
        
        words_size= len(words_dic)
        words_redic = dict(zip(words_dic, range(words_size))) #反向字典
        print(words_redic)
        print('字表大小:', words_size)
        return words_dic,words_redic

    def wordtov(self,words_redic,word) :
        if words_redic.has_key(word):
            return words_redic[word]
        else:
            return 0 # 字典里没有的就是None
               
    #字符到向量
    def ch_to_v(self,datalist,words_redic,normal = 1):
        
        to_num = lambda word: words_redic[word] if word in words_redic else 0# 字典里没有的就是None

        data_vector =[]
        for ii in datalist:
            one_vector = list(map(to_num, list(ii))) 
            data_vector.append(one_vector)         
        #归一化
        if normal == 1:
            return np.asarray(data_vector)/ (len(words_redic)/2) - 1 
        return np.array(data_vector)
        

    def pad_sequences(self,sequences, maxlen=None, dtype=np.float32,
                      padding='post', truncating='post', value=0.):
    
        
        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    
        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)
    
        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
    
        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)
    
            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))
    
            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x, lengths



                
   