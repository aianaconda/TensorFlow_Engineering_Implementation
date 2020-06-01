# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import tensorflow as tf
import preprocessing 

positive_data_file ="./data/rt-polaritydata/rt-polarity.pos"
negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"

def mydataset(positive_data_file,negative_data_file):  #定义函数创建数据集
    filelist = [positive_data_file,negative_data_file]
    
    def gline(filelist):                                #定义生成器函数，返回每一行的数据
        for file in filelist:
            with open(file, "r",encoding='utf-8') as f:
                for line in f:
                    yield line
                    
    x_text = gline(filelist)
    lenlist = [len(x.split(" ")) for x in x_text]
    max_document_length = max(lenlist)
    vocab_processor = preprocessing.VocabularyProcessor(max_document_length,5)
    
    x_text = gline(filelist)
    vocab_processor.fit(x_text)
    a=list (vocab_processor.reverse( [list(range(0,len(vocab_processor.vocabulary_)))] ))
    print("字典：",a)
    
    def gen():  #循环生成器（不然一次生成器结束就会没有了）
        while True:
            x_text2 = gline(filelist)
            for i ,x in enumerate(vocab_processor.transform(x_text2)):
                if i < int(len(lenlist)/2):
                    onehot = [1,0]
                else:
                    onehot = [0,1]
                yield (x,onehot)
    
    data = tf.data.Dataset.from_generator( gen,(tf.int64,tf.int64) )
    data = data.shuffle(len(lenlist))
    data = data.batch(256)
    data = data.prefetch(1)
    return data,vocab_processor,max_document_length  #返回数据集、字典、最大长度

if __name__ == '__main__':                                      #单元测试代码
    data,_,_ =mydataset(positive_data_file,negative_data_file)
    iterator = data.make_initializable_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess2:
      sess2.run(iterator.initializer)
      for i in range(80):
          print("batched data 1:",i)#,
          sess2.run(next_element)
