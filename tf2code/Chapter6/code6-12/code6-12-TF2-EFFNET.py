# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions,preprocess_input
from keras_efficientnets import EfficientNetB0
#from keras_efficientnets import EfficientNetB4


#6.7.10节 动态图
#tf.enable_eager_execution()						#启动动态图
#model = EfficientNetB4(weights='efficientnet-b4.h5')
model = EfficientNetB0(weights='efficientnet-b0.h5')

img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

#6.7.10节 动态图
#preds = model(x)
#print('Predicted:', decode_predictions(preds.numpy(), top=3)[0])