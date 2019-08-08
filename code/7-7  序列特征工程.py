# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf

tf.reset_default_graph()

vocabulary_size = 3                           #假如有3个词，向量为0，1，2

sparse_input_a = tf.SparseTensor(             #定义一个稀疏矩阵,  值为：  
    indices=((0, 0), (1, 0), (1, 1)),         #      [2]   只有一个序列
    values=(2, 0, 1),                         #      [0, 1] 有连个序列
    dense_shape=(2, 2))

sparse_input_b = tf.SparseTensor(             #定义一个稀疏矩阵,  值为：  
    indices=((0, 0), (1, 0), (1, 1)),         #      [1]
    values=(1, 2, 0),                         #      [2, 0]
    dense_shape=(2, 2))

embedding_dimension_a = 2
embedding_values_a = (                      #为稀疏矩阵的三个值（0，1，2）匹配词嵌入初始值
    (1., 2.),  # id 0
    (3., 4.),  # id 1
    (5., 6.)  # id 2
)
embedding_dimension_b = 3
embedding_values_b = (                     #为稀疏矩阵的三个值（0，1，2）匹配词嵌入初始值
    (11., 12., 13.),  # id 0
    (14., 15., 16.),  # id 1
    (17., 18., 19.)  # id 2
)

def _get_initializer(embedding_dimension, embedding_values): #自定义初始化词嵌入
  def _initializer(shape, dtype, partition_info):
    return embedding_values
  return _initializer



categorical_column_a = tf.contrib.feature_column.sequence_categorical_column_with_identity( #带序列的离散列
    key='a', num_buckets=vocabulary_size)
embedding_column_a = tf.feature_column.embedding_column(    #将离散列转为词向量
    categorical_column_a, dimension=embedding_dimension_a,
    initializer=_get_initializer(embedding_dimension_a, embedding_values_a))


categorical_column_b = tf.contrib.feature_column.sequence_categorical_column_with_identity(
    key='b', num_buckets=vocabulary_size)
embedding_column_b = tf.feature_column.embedding_column(
    categorical_column_b, dimension=embedding_dimension_b,
    initializer=_get_initializer(embedding_dimension_b, embedding_values_b))



shared_embedding_columns = tf.feature_column.shared_embedding_columns( #共享列
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension_a,
        initializer=_get_initializer(embedding_dimension_a, embedding_values_a))
 
features={                                              #将a,b合起来
        'a': sparse_input_a,
        'b': sparse_input_b,
    }

input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(    #定义序列输入层
        features,
    feature_columns=[embedding_column_b, embedding_column_a])

input_layer2, sequence_length2 = tf.contrib.feature_column.sequence_input_layer(    #定义序列输入层
        features,
    feature_columns=shared_embedding_columns)

global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)#返回图中的张量（2个嵌入词权重）
print([v.name for v in global_vars])  

with tf.train.MonitoredSession() as sess:
    print(global_vars[0].eval(session=sess))#输出词向量的初始值
    print(global_vars[1].eval(session=sess))
    print(global_vars[2].eval(session=sess))
    print(sequence_length.eval(session=sess))
    print(input_layer.eval(session=sess))   #输出序列输入层的内容
    print(sequence_length2.eval(session=sess))  
    print(input_layer2.eval(session=sess))   #输出序列输入层的内容

    
