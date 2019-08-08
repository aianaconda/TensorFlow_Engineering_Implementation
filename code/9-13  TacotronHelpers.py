"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper

def _go_frames(batch_size, output_dim):#输入的目标序列以0开始。做为<GO>标志
  return tf.tile([[0.0]], [batch_size, output_dim])


class TacoTrainingHelper(Helper):#训练场景下的采样接口

  def __init__(self,  targets, output_dim, r): #targets形状为[N, T_out, D]

    with tf.name_scope('TacoTrainingHelper'):
      self._batch_size = tf.shape(targets)[0] #获得批次
      self._output_dim = output_dim
      self._reduction_factor = r

      #对目标输入进行步长为r的采样。每r(5)个mel中取一个作为下一时刻的y
      self._targets = targets[:, r-1::r, :]

      num_steps = tf.shape(self._targets)[1] #获得序列长度（采样后的最大步数）
      self._lengths = tf.tile([num_steps], [self._batch_size])#构建RNN输入所用的长度矩阵

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def token_output_size(self):#输出的大小为5
    return self._reduction_factor

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):#初始时设置全0输入，代表go
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None): #只是补充接口，会输入到在next_inputs中的sample_ids
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, time, outputs, state,  name=None, **unused_kwargs):#取time时刻的数据传入
    with tf.name_scope(name or 'TacoTrainingHelper'):
      finished = (time + 1 >= self._lengths)  #判断是否结束
      next_inputs = self._targets[:, time, :] # Teacher forcing: feed the true frame
      return (finished, next_inputs, state)

class TacoTestHelper(Helper): #测试场景下的采样接口
  def __init__(self, batch_size, output_dim, r):
    with tf.name_scope('TacoTestHelper'):
      self._batch_size = batch_size
      self._output_dim = output_dim
      self._reduction_factor = r  #采样的步长

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def token_output_size(self):  #自定义属性
    return self._reduction_factor

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, outputs, state,  stop_token_preds, name=None, **unused_kwargs):#测试时靠stop_token_preds判断结束

    with tf.name_scope('TacoTestHelper'):
      #如果stop概率> 0.5，即为stop标志
      finished = tf.reduce_any(tf.cast(tf.round(stop_token_preds), tf.bool))

      #将解码器输出的最后一帧作为下一时刻的输入
      next_inputs = outputs[:, -self._output_dim:]
      return (finished, next_inputs, state)
