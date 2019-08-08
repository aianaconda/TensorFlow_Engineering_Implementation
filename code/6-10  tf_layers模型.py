# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.VERSION))
#print("Eager execution: {}".format(tf.executing_eagerly()))
#tf.logging.set_verbosity (tf.logging.ERROR)
# 加载训练和验证数据集

import tensorflow_datasets as tfds


ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"]) #加载数据集
ds_train = ds_train.shuffle(1000).batch(10).prefetch(tf.data.experimental.AUTOTUNE)#用tf.data.Dataset接口加工数据集

class MNISTModel(tf.layers.Layer):
  def __init__(self, name):
    super(MNISTModel, self).__init__(name=name)

    self._input_shape = [-1, 28, 28, 1]
    self.conv1 =  tf.layers.Conv2D(32, 5,  activation=tf.nn.relu)
    self.conv2 =  tf.layers.Conv2D(64, 5,  activation=tf.nn.relu)
    self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
    self.fc2 = tf.layers.Dense(10)
    self.dropout = tf.layers.Dropout(0.5)
    self.max_pool2d =  tf.layers.MaxPooling2D(
            (2, 2), (2, 2), padding='SAME')

  def call(self, inputs, training):
    x = tf.reshape(inputs, self._input_shape)
    x = self.conv1(x)
    x = self.max_pool2d(x)
    x = self.conv2(x)
    x = self.max_pool2d(x)
    x = tf.keras.layers.Flatten()(x)
    x = self.fc1(x)
    if training:
      x = self.dropout(x)
    x = self.fc2(x)
    return x

def loss(model,inputs, labels):
    predictions = model(inputs, training=True)
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=predictions, labels=labels )
    return tf.reduce_mean( cost )
# 训练
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
grad = tfe.implicit_gradients(loss)


model = MNISTModel("net")

global_step = tf.train.get_or_create_global_step()

for epoch in range(1):
    for i,data in enumerate(ds_train):
        inputs, targets =tf.cast( data["image"],tf.float32), data["label"]

        optimizer.apply_gradients(grad( model,inputs, targets), global_step=global_step)

        if i % 400 == 0:
          print("Step %d: Loss on training set : %f" %
                (i, loss(model,inputs, targets).numpy()))


          all_variables = (
              model.variables
              + optimizer.variables()
              + [global_step])
          tfe.Saver(all_variables).save(
              "./log/linermodel.cpkt", global_step=global_step)
ds = tfds.as_numpy(ds_test.batch(100))
onetestdata = next(ds)

print("Loss on test set: %f" % loss( model,onetestdata["image"].astype(np.float32), onetestdata["label"]).numpy())
