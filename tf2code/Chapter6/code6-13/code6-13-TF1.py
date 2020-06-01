# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions


inputs = tf.placeholder(tf.float32, (224, 224, 3))#定义占位符

tensorimg = tf.expand_dims(inputs, 0)               #预处理
tensorimg =preprocess_input(tensorimg)




with tf.Session() as sess:  #在session中运行
    sess.run(tf.global_variables_initializer())
    Reslayer = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    logits = Reslayer(tensorimg)  #网络模型

    #Reslayer = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5',
    #                 input_tensor=tensorimg,input_shape = (224, 224, 3) )
    #logits = Reslayer.layers[-1].output
    #print(logits)

    img_path = './dog.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    logitsv = sess.run(logits,feed_dict={inputs: img})
    Pred =decode_predictions(logitsv, top=3)[0]
    print('Predicted:', Pred,len(logitsv[0]))

#可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
fig.sca(ax1)
ax1.imshow(img)
fig.sca(ax1)

barlist = ax2.bar(range(3), [ i[2] for i in Pred ])
barlist[0].set_color('g')

plt.sca(ax2)
plt.ylim([0, 1.1])
#plt.xticks(range(3),[i[1][:15] for i in Pred], rotation='vertical')
fig.subplots_adjust(bottom=0.2)
plt.show()


