# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import os
import numpy as np


_pad,_eos = '_','~'#定义填充字符与结束字符
_padv = 0 #定义填充的向量占位符
_stop_token_padv = 1#定义标志结束的向量占位符


_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\'(),-.:;? '
symbols = [_pad, _eos] + list(_characters)#定义字典

index_symbols = {value:key for key, value in enumerate(symbols) }
print(index_symbols)

def sequence_to_text(sequence):#将向量转成字符
    strlen = len(symbols)
    return ''.join([symbols[i] for i in sequence if i<strlen ])




#定义数据集
def mydataset(metadata_filename,outputs_per_step,batch=32,shuffleflag=True,mode = 'train'):

    #加载metadata文件
    datadir = os.path.dirname(metadata_filename)
    with open(metadata_filename, encoding='utf-8') as f:
        print("metadata_filename",metadata_filename)
        _metadata = [line.strip().split('|') for line in f]

    #加载拼音字符，并计算最大长度
    inputseq = list( map(lambda x:[index_symbols[key] for key in  x[3] ],_metadata) )
    seqlen = [len(x) for x in inputseq]

    #计算语音最大长度
    Max_output_length =int(max(m[2] for m in _metadata))+1
    #对长度按照outputs_per_step步长，进行取整
    Max_output_length =[ Max_output_length + outputs_per_step - Max_output_length%outputs_per_step,Max_output_length][Max_output_length%outputs_per_step==0]

    #对输入拼音补0
    inputseq = tf.keras.preprocessing.sequence.pad_sequences(inputseq, padding='post',value=_padv)
    print(inputseq)
    print(len(inputseq[0]))
    #定义拼音数据集
    datasetinputseq = tf.data.Dataset.from_tensor_slices( inputseq )
    #定义输入长度数据集
    datasetseqlen = tf.data.Dataset.from_tensor_slices( seqlen )
    #定义全部的metadata数据集
    datasetmetadata = tf.data.Dataset.from_tensor_slices( _metadata )
    #合并数据集
    dataset = tf.data.Dataset.zip((datasetmetadata,datasetinputseq,datasetseqlen))
    print(dataset.output_shapes)

    def mymap(_meta,seq,seqlen):#对合并好的数据集按照指定规则进行处理
        def _parse(meta):
            #根据文件名，加载音频数据的np文件
            linear_target = np.load(os.path.join(datadir, meta.numpy()[0].decode('UTF-8') ))
            mel_target = np.load(os.path.join(datadir, meta.numpy()[1].decode('UTF-8')))

            #构造结束掩码
            stop_token_target = np.asarray([0.] * len(mel_target),dtype = np.float32)

            #统一对齐操作
            linear_target =np.pad(linear_target, [(0, Max_output_length - linear_target.shape[0]), (0,0)], mode='constant', constant_values=_padv)
            mel_target =np.pad(mel_target, [(0, Max_output_length - mel_target.shape[0]), (0,0)], mode='constant', constant_values=_padv)
            stop_token_target =np.pad(stop_token_target, (0, Max_output_length - len(stop_token_target)), mode='constant', constant_values=_stop_token_padv)
            #返回处理后的单条样本
            return linear_target,mel_target,stop_token_target
        linear_target,mel_target,stop_token_target = tf.py_function( _parse, [_meta], [tf.float32,tf.float32,tf.float32])
        return seq,seqlen,linear_target,mel_target,stop_token_target#调用第三方函数进行map处理的返回值

    dataset = dataset.map(mymap)	#对数据进行map处理

    if mode=='train':#训练场景下进行乱序
        if shuffleflag == True:   #对数据进行乱序操作
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch) #批次划分

    iterator = dataset.make_one_shot_iterator()#创建生成器。取一条
    print(dataset.output_types)
    print(dataset.output_shapes)
    next_element = iterator.get_next()
    return next_element


if __name__ == '__main__':

    ##测试代码
    next_element = mydataset('training/train.txt',5)
    with tf.Session() as sess2:
      print("batched data 1:",sess2.run(next_element))

    #a = [34,45,34,6,7,5,6,234,3,4,5,6]
    #print(sequence_to_text(a))
    #input = [[1,2], [2,2], [3,2]]
    #padding = [[0,4],[0,0]]
    #with tf.Session() as sess :
    #    print(sess.run(tf.pad(input,padding,constant_values = 1)))