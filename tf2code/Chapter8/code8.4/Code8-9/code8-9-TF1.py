# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import tensorflow as tf
import numpy as np
#tf.enable_eager_execution()
attention_keras = __import__("8-10  keras注意力机制模型")

#定义参数
num_words = 20000
maxlen = 80
batch_size = 32

#加载数据
print('Loading data...')
(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.imdb.load_data(path='./imdb.npz',num_words=num_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(x_train[0])
print(y_train[:10])

word_index = tf.keras.datasets.imdb.get_word_index('./imdb_word_index.json')# 单词--下标 对应字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])# 下标-单词对应字典

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
print(decoded_newswire)



#数据对齐
x_train =  tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test =  tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print('Pad sequences x_train shape:', x_train.shape)

#定义输入节点
S_inputs = tf.keras.layers.Input(shape=(None,), dtype='int32')

#生成词向量
embeddings = tf.keras.layers.Embedding(num_words, 128)(S_inputs)
embeddings = attention_keras.Position_Embedding()(embeddings) #默认使用同等维度的位置向量



#使用内部注意力模型处理
O_seq = attention_keras.Attention(8,16)([embeddings,embeddings,embeddings])
print("O_seq",O_seq)
#将结果进行全局池化
O_seq = tf.keras.layers.GlobalAveragePooling1D()(O_seq)
#添加dropout
#O_seq = tf.keras.layers.Dropout(0.5)(O_seq)
O_seq =attention_keras.TargetedDropout(drop_rate=0.5, target_rate=0.5)(O_seq)
#输出最终节点
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(O_seq)
print(outputs)
#将网络结构组合到一起
model = tf.keras.models.Model(inputs=S_inputs, outputs=outputs)

#添加反向传播节点
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#开始训练
print('Train...')
model.fit(x_train, y_train, batch_size=batch_size,epochs=5, validation_data=(x_test, y_test))