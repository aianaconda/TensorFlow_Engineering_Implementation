# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:13:28 2018

@author: ljh
"""

import tensorflow as tf
import numpy as np

#测试自定义hook
x = tf.compat.v1.placeholder(shape=(10, 2), dtype=tf.float32)
w = tf.Variable(initial_value=[[10.], [10.]])
w0 = [[1], [1.]]
y = tf.matmul(x, w0)
loss = tf.reduce_mean(input_tensor=(tf.matmul(x, w) - y) ** 2)
optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

class _Hook(tf.estimator.SessionRunHook):
  def __init__(self, loss):
    self.loss = loss

  def begin(self):
    pass

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self.loss)

  def after_run(self, run_context, run_values):
    loss_value = run_values.results
    print("loss value:", loss_value)

sess = tf.compat.v1.train.MonitoredSession(hooks=[_Hook(loss)])
for _ in range(10):
  x_ = np.random.random((10, 2))
  sess.run(optimizer, {x: x_})


tf.compat.v1.reset_default_graph()
#测试FeedFnHook
x2 = tf.compat.v1.placeholder(dtype=tf.float32)
y = x2 + 1
hook = tf.estimator.FeedFnHook(feed_fn=lambda: {x2: 1.0})
sess = tf.compat.v1.train.MonitoredSession(hooks=[hook])
print(sess.run(y))







