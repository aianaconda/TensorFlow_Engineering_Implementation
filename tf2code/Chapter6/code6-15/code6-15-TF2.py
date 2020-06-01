# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import numpy as np
tf.compat.v1.disable_v2_behavior()
#在内存中生成模拟数据
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X为-1到1之间连续的100个浮点数
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
    return train_X, train_Y   #以生成器的方式返回

train_data = GenerateData()  
test_data = GenerateData(20)  
batch_size=10

def train_input_fn(train_data, batch_size):  #定义训练数据集输入函数
    #构造数据集的组成：一个特征输入，一个标签输入
    dataset = tf.data.Dataset.from_tensor_slices( (  train_data[0],train_data[1]) )   
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #将数据集乱序、重复、批次划分. 
    return dataset     #返回数据集 

def eval_input_fn(data,labels, batch_size):  #定义测试或应用模型时，数据集的输入函数
    #batch不允许为空
    assert batch_size is not None, "batch_size must not be None" 
    
    if labels is None:  #如果评估，则没有标签
        inputs = data  
    else:  
        inputs = (data,labels)  
    #构造数据集 
    dataset = tf.data.Dataset.from_tensor_slices(inputs)  
 
    dataset = dataset.batch(batch_size)  #按批次划分
    return dataset     #返回数据集     

def my_model(features, labels, mode, params):#自定义模型函数：参数是固定的。一个特征，一个标签
    #定义网络结构
    W = tf.Variable(tf.random.normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    # 前向结构
    predictions = tf.multiply(tf.cast(features,dtype = tf.float32), W)+ b
    
    if mode == tf.estimator.ModeKeys.PREDICT: #预测处理
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    #定义损失函数
    loss = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=predictions)

    meanloss  = tf.compat.v1.metrics.mean(loss)#添加评估输出项
    metrics = {'meanloss':meanloss}

    if mode == tf.estimator.ModeKeys.EVAL: #测试处理
        return tf.estimator.EstimatorSpec(   mode, loss=loss, eval_metric_ops=metrics) 
        #return tf.estimator.EstimatorSpec(   mode, loss=loss)

    #训练处理.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)  


tf.compat.v1.reset_default_graph()  #清空图
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)      #能够控制输出信息  ，
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)  #构建gpu_options，防止显存占满
session_config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
#构建估算器
estimator = tf.estimator.Estimator(  model_fn=my_model,model_dir='myestimatormode',params={'learning_rate': 0.1},
                                   config=tf.estimator.RunConfig(session_config=session_config)  )


train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_data, batch_size), max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(test_data,None, batch_size))

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)








