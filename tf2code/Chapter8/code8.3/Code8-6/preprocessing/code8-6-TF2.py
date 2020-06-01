# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.compat.v1.disable_v2_behavior()
positive_data_file ="./data/rt-polaritydata/rt-polarity.pos"
negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"

def mydataset(positive_data_file,negative_data_file):  #定义函数创建数据集
    filelist = [positive_data_file,negative_data_file]
    
    def gline(filelist):                                #定义生成器函数，返回每一行的数据
        for file in filelist:
            with open(file, "r",encoding='utf-8') as f:
                for line in f:
                    yield line
                    
    x_text_= gline(filelist)
    x_text_raw = gline(filelist)
    x_text=[]
    while True:
        try:
            x_text.extend(next(x.split(" ") for x in x_text_))
        except StopIteration:
            break
    lenlist = [len(x.split(" ")) for x in x_text_raw]
    max_document_length = max(lenlist)#61        
    tokenizer = Tokenizer(num_words=20000,char_level=False,oov_token='<UNK>')
    tokenizer.fit_on_texts(x_text)#必须的
    train_idxs = tokenizer.texts_to_sequences(x_text)
    wordindex = tokenizer.word_index
    
    a=list (list(wordindex.keys()))
    print("字典：",a)
    
    def gen():  #循环生成器（不然一次生成器结束就会没有了）
        while True:
            x_text2 = gline(filelist)
            train_idxs = tokenizer.texts_to_sequences(x_text2)
            train_padded = pad_sequences(train_idxs,maxlen=max_document_length, padding='post', truncating='post')
            for i ,x in enumerate(train_padded):
                if i < int(len(lenlist)/2):
                    onehot = [1,0]
                else:
                    onehot = [0,1]
                yield (x,onehot)#onehot 是预测的结果，1表示positive，0表示negative。
    
    data = tf.data.Dataset.from_generator(gen,(tf.int64,tf.int64) )
    data = data.shuffle(len(lenlist))
    data = data.batch(256)
    data = data.prefetch(1)
    return data,tokenizer,max_document_length#返回数据集、字典、最大长度

if __name__ == '__main__':                                      #单元测试代码
    data,_,_ =mydataset(positive_data_file,negative_data_file)
    iterator = tf.compat.v1.data.make_initializable_iterator(data)
    next_element = iterator.get_next()
    
    with tf.compat.v1.Session() as sess2:
      sess2.run(iterator.initializer)
      for i in range(80):
          print("batched data 1:",i)#,
          sess2.run(next_element)