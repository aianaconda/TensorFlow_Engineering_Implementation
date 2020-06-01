# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
tf.compat.v1.disable_v2_behavior()#加不加这句都可以
#将离散文本按照指定范围散列
def test_categorical_cols_to_hash_bucket():
    tf.compat.v1.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)#稀疏矩阵，单独放进去会出错
   
    builder = _LazyBuilder({
          'sparse_feature': [['a'], ['x']],
      })
    id_weight_pair = some_sparse_column._get_sparse_tensors(builder) #

    with tf.compat.v1.Session() as sess:
        
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print("稀疏矩阵：\n",id_tensor_eval)
          
        dense_decoded = tf.sparse.to_dense( id_tensor_eval, default_value=-1).eval(session=sess)
        print("稠密矩阵：\n",dense_decoded)

test_categorical_cols_to_hash_bucket()


from tensorflow.python.ops import lookup_ops
#将离散文本按照指定词表与指定范围，混合散列
def test_with_1d_sparse_tensor():
    tf.compat.v1.reset_default_graph()
    #混合散列
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        'name', vocabulary_list=['anna', 'gary', 'bob'],num_oov_buckets=2)   #稀疏矩阵
    
    #稠密矩阵
    builder = _LazyBuilder({
        'name': ['anna', 'gary','alsa'],        
      })
    
    #稀疏矩阵
    builder2 = _LazyBuilder({
        'name': tf.SparseTensor(
        indices=((0,), (1,), (2,)),
        values=('anna', 'gary', 'alsa'),
        dense_shape=(3,)),    
      })    

    id_weight_pair = body_style._get_sparse_tensors(builder)    #
    id_weight_pair2 = body_style._get_sparse_tensors(builder2)  #


    with tf.compat.v1.Session() as sess:
        sess.run(lookup_ops.tables_initializer())

        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print("稀疏矩阵：\n",id_tensor_eval)
        id_tensor_eval2 = id_weight_pair2.id_tensor.eval()
        print("稀疏矩阵2：\n",id_tensor_eval2)
          
        dense_decoded = tf.sparse.to_dense( id_tensor_eval, default_value=-1).eval(session=sess)
        print("稠密矩阵：\n",dense_decoded)

test_with_1d_sparse_tensor()


#将离散文本转为onehot特征列
def test_categorical_cols_to_onehot():
    tf.compat.v1.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)                                       #定义散列特征列
    
    #转换成one-hot特征列
    one_hot_style = tf.feature_column.indicator_column(some_sparse_column)     
  

    features = {
      'sparse_feature': [['a'], ['x']],
      }

    net = tf.compat.v1.feature_column.input_layer(features, one_hot_style)               #生成输入层张量
    with tf.compat.v1.Session() as sess:                                                      #通过会话输出数据
        print(net.eval()) 

test_categorical_cols_to_onehot()





#将离散文本转为onehot词嵌入特征列
def test_categorical_cols_to_embedding():
    tf.compat.v1.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)#稀疏矩阵，单独放进去会出错
   
    embedding_col = tf.feature_column.embedding_column( some_sparse_column, dimension=3)

    features = {
          'sparse_feature': [['a'], ['x']],
      }
    
    #生成输入层张量
    cols_to_vars = {}
    net = tf.compat.v1.feature_column.input_layer(features, embedding_col,cols_to_vars)
  
    with tf.compat.v1.Session() as sess:                  #通过会话输出数据
        sess.run(tf.compat.v1.global_variables_initializer())
        print(net.eval()) 

test_categorical_cols_to_embedding()

#input_layer中的顺序
def test_order():
    tf.compat.v1.reset_default_graph()
    numeric_col = tf.feature_column.numeric_column('numeric_col')
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'asparse_feature', hash_bucket_size=5)#稀疏矩阵，单独放进去会出错
   
    embedding_col = tf.feature_column.embedding_column( some_sparse_column, dimension=3)
    #转换成one-hot特征列
    one_hot_col = tf.feature_column.indicator_column(some_sparse_column)
    print(one_hot_col.name)
    print(embedding_col.name)
    print(numeric_col.name)

    features = {
          'numeric_col': [[3], [6]],
          'asparse_feature': [['a'], ['x']],
      }
    
    #生成输入层张量
    cols_to_vars = {}
    net = tf.compat.v1.feature_column.input_layer(features, [numeric_col,one_hot_col,embedding_col],cols_to_vars)

    with tf.compat.v1.Session() as sess:                  #通过会话输出数据
        sess.run(tf.compat.v1.global_variables_initializer())
        print(net.eval()) 

test_order()
