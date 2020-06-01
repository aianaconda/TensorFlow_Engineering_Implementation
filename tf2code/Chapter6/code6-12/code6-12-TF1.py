# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
#6.7.10节 动态图
#tf.enable_eager_execution()						#启动动态图

model = ResNet50(weights='imagenet')
#model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')


img_path = 'hy.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

#6.7.10节 动态图
#preds = model(x)
#print('Predicted:', decode_predictions(preds.numpy(), top=3)[0])