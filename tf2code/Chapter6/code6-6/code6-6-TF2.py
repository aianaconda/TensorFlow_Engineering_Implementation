# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 11:04:32 2018

@author: ljh
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
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)  #构建gpu_options，防止显存占满
session_config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
#构建估算器
estimator = tf.estimator.Estimator(  model_fn=my_model,model_dir='./myestimatormode',params={'learning_rate': 0.1},
                                   config=tf.estimator.RunConfig(session_config=session_config)  )
#匿名输入方式
estimator.train(lambda: train_input_fn(train_data, batch_size),steps=200)
tf.compat.v1.logging.info("训练完成.")#输出训练完成
##偏函数方式
#from functools import partial
#estimator.train(input_fn=partial(train_input_fn, train_data=train_data, batch_size=batch_size),steps=2)
#
##装饰器方式
#def checkParams(fn):					#定义通用参数装饰器函数
#    def wrapper():			#使用字典和元组的解包参数来作形参
#        return fn(train_data=train_data, batch_size=batch_size)           	#如满足条件，则将参数透传给原函数，并返回
#    return wrapper
#
#@checkParams
#def train_input_fn2(train_data, batch_size):  #定义训练数据集输入函数
#    #构造数据集的组成：一个特征输入，一个标签输入
#    dataset = tf.data.Dataset.from_tensor_slices( (  train_data[0],train_data[1]) )   
#    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #将数据集乱序、重复、批次划分. 
#    return dataset     #返回数据集 
#estimator.train(input_fn=train_input_fn2, steps=2)
#
#tf.logging.info("训练完成.")#输出训练完成




#热启动
warm_start_from = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from='./myestimatormode',
        )
#重新定义带有热启动的估算器
estimator2 = tf.estimator.Estimator(  model_fn=my_model,model_dir='./myestimatormode3',warm_start_from=warm_start_from,params={'learning_rate': 0.1},
                                   config=tf.estimator.RunConfig(session_config=session_config)  )
estimator2.train(lambda: train_input_fn(train_data, batch_size),steps=200)

test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(test_data[0],test_data[1],batch_size=1,shuffle=False)
train_metrics = estimator.evaluate(input_fn=test_input_fn)
#train_metrics = estimator2.evaluate(input_fn=lambda: eval_input_fn(train_data[0],train_data[1],batch_size))
#
print("train_metrics",train_metrics)
#
predictions = estimator.predict(input_fn=lambda: eval_input_fn(test_data[0],None,batch_size))
print("predictions",list(predictions))

new_samples = np.array( [6.4, 3.2, 4.5, 1.5], dtype=np.float32)    #定义输入
predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(  new_samples,num_epochs=1, batch_size=1,shuffle=False)
predictions = list(estimator.predict(input_fn=predict_input_fn))
print( "输入, 结果:  {}  {}\n".format(new_samples,predictions))