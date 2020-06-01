# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
tf.compat.v1.disable_v2_behavior()
def test_crossed():   
    
    a = tf.feature_column.numeric_column('a', dtype=tf.int32, shape=(2,))
    b = tf.feature_column.bucketized_column(a, boundaries=(0, 1))               #离散值转化    
    crossed = tf.feature_column.crossed_column([b, 'c'], hash_bucket_size=5)#生成交叉列

    builder = _LazyBuilder({                                                #生成模拟输入的数据
          'a':
              tf.constant(((-1.,-1.5), (.5, 1.))),
          'c':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      })
    id_weight_pair = crossed._get_sparse_tensors(builder)#生成输入层张量      
    with tf.compat.v1.Session() as sess2:                             #建立会话session，取值
          id_tensor_eval = id_weight_pair.id_tensor.eval()
          print(id_tensor_eval)                             #输出稀疏矩阵
          
          dense_decoded = tf.sparse.to_dense( id_tensor_eval, default_value=-1).eval(session=sess2)
          print(dense_decoded)                               #输出稠密矩阵
          
test_crossed()

