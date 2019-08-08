# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import numpy as np

import tensorflow as tf
import pandas as pd
from matplotlib import pyplot

tf.logging.set_verbosity(tf.logging.INFO)

csv_file_name = './number-of-daily-births-in-quebec.csv'
md1 = pd.read_csv(csv_file_name,names=list('AB'),skiprows=1,encoding = "gbk") #,skiprows=1,columns=list('ABCD')                   

data_num=np.array(md1["B"])
print(data_num[:10])

x = np.array(range(len(data_num)))
data = {
    tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES: data_num,
}

reader = tf.contrib.timeseries.NumpyReader(data)

estimator = tf.contrib.timeseries.StructuralEnsembleRegressor(#定义TFTS算器
  periodicities=200, num_features=1, cycle_num_latent_values=15,model_dir ="mode/")#耗内存，但是准确

#定义输入函数
train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=4, window_size=64)
  
estimator.train(input_fn=train_input_fn, steps=600)#训练模型 
  

evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)   
evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)

print(evaluation.keys())
print(evaluation['loss'])

(predictions,) = tuple(estimator.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=2)))
print("predictions:",predictions)


times = evaluation["times"][0][-20:]   #取后20个进行显示
observed = evaluation["observed"][0, :, 0][-20:]
mean = np.squeeze(np.concatenate(
      [evaluation["mean"][0][-20:], predictions["mean"]], axis=0))
variance = np.squeeze(np.concatenate(
      [evaluation["covariance"][0][-20:], predictions["covariance"]], axis=0))
all_times = np.concatenate([times, predictions["times"]], axis=0)
upper_limit = mean + np.sqrt(variance)
lower_limit = mean - np.sqrt(variance)

def make_plot(name, training_times, observed, all_times, mean, upper_limit, lower_limit):
  """Plot a time series in a new figure."""
  pyplot.figure()
  pyplot.plot(training_times, observed, "b", label="training series")
  pyplot.plot(all_times, mean, "r", label="forecast")
  pyplot.plot(all_times, upper_limit, "g", label="forecast upper bound")
  pyplot.plot(all_times, lower_limit, "g", label="forecast lower bound")
  pyplot.fill_between(all_times, lower_limit, upper_limit, color="grey",
                      alpha="0.2")
  pyplot.axvline(training_times[-1], color="k", linestyle="--")
  pyplot.xlabel("time")
  pyplot.ylabel("observations")
  pyplot.legend(loc=0)
  pyplot.title(name)
  
make_plot("Structural ensemble",times,observed,all_times,mean,upper_limit, lower_limit)

