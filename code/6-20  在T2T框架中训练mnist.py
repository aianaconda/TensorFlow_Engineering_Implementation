# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import metrics

#启动动态图
tfe = tf.contrib.eager
tf.enable_eager_execution()

problems.available()



#建立路径
data_dir = os.path.expanduser("./t2t/data")
tmp_dir = os.path.expanduser("./t2t/tmp")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)

#下载数据集
mnist_problem = problems.problem("image_mnist")
mnist_problem.generate_data(data_dir, tmp_dir)#下载到tmp_dir，（分为训练和测试）转换后存到data_dir

#取出一个数据，并显示
Modes = tf.estimator.ModeKeys
mnist_example = tfe.Iterator(mnist_problem.dataset(Modes.TRAIN, data_dir)).next()
image = mnist_example["inputs"]#一个数据集元素的张量
label = mnist_example["targets"]

plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap('gray'))
print("Label: %d" % label.numpy())

print(Modes)

#registry.list_models()
#registry.list_hparams("transformer")


class MySimpleModel(t2t_model.T2TModel):#自定义模型

  def body(self, features):             #实现body方法
    inputs = features["inputs"]
    filters = self.hparams.hidden_size
    #h1=(in_width–filter_width + 1) / strides_ width =[12*12]
    h1 = tf.layers.conv2d(inputs, filters, kernel_size=(5, 5), strides=(2, 2))#默认valid
    #h2=[4*4]
    h2 = tf.layers.conv2d(tf.nn.relu(h1), filters, kernel_size=(5, 5), strides=(2, 2))

    return tf.layers.conv2d(tf.nn.relu(h2), filters, kernel_size=(3, 3))#[1*1]

hparams = trainer_lib.create_hparams("basic_1", data_dir=data_dir, problem_name="image_mnist")
hparams.hidden_size = 64
model = MySimpleModel(hparams, Modes.TRAIN)


#使用装饰器implicit_value_and_gradients，来封装loss函数
@tfe.implicit_value_and_gradients
def loss_fn(features):
  _, losses = model(features)
  return losses["training"]


BATCH_SIZE = 128        #指定批次
#创建数据集
mnist_train_dataset = mnist_problem.dataset(Modes.TRAIN, data_dir)
mnist_train_dataset = mnist_train_dataset.repeat(None).batch(BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()#定义优化器



#训练模型
NUM_STEPS = 500#指定训练次数
for count, example in enumerate(mnist_train_dataset):
  example["targets"] = tf.reshape(example["targets"], [BATCH_SIZE, 1, 1, 1])  # Make it 4D.
  loss, gv = loss_fn(example)
  optimizer.apply_gradients(gv)

  if count % 50 == 0:
    print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
  if count >= NUM_STEPS:
    break
#######
model.set_mode(Modes.EVAL)
mnist_eval_dataset = mnist_problem.dataset(Modes.EVAL, data_dir) #定义评估数据集

#创建评估metrics，返回准确率与top5的准确率。
metrics_accum, metrics_result = metrics.create_eager_metrics(
    [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5])

for count, example in enumerate(mnist_eval_dataset):#遍历数据
  if count >= 200:#只取200个
    break

  #变化形状
  example["inputs"] = tf.reshape(example["inputs"], [1, 28, 28, 1])
  example["targets"] = tf.reshape(example["targets"], [1, 1, 1, 1])

  predictions, _ = model(example)#用模型计算

  #计算统计值
  metrics_accum(predictions, example["targets"])

# Print out the averaged metric values on the eval data
for name, val in metrics_result().items():
  print("%s: %.2f" % (name, val))





















