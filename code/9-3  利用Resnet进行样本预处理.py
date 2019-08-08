# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""



import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import ResNet50

def makenumpyfeature(numpyPATH,img_filename,PATH):
    if os.path.exists(numpyPATH):
        shutil.rmtree(numpyPATH, ignore_errors=True)
    os.mkdir(numpyPATH)


    size = [224,224]
    batchsize = 10

    def load_image(image_path):
        img = tf.read_file(PATH +image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, image_path


    image_model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
                     ,include_top=False)#创建ResNet网络

    new_input = image_model.input
    hidden_layer = image_model.layers[-2].output #获取ResNet的倒数第二层（池化前的卷积结果）

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    #对文件目录去重
    encode_train = sorted(set(img_filename))

    #图片数据集
    image_dataset = tf.data.Dataset.from_tensor_slices(
                                    encode_train).map(load_image).batch(batchsize)

    for img, path in image_dataset:
      batch_features = image_features_extract_model(img)
      print(batch_features.shape)
      batch_features = tf.reshape(batch_features,
                                  (batch_features.shape[0], -1, batch_features.shape[3]))
      print(batch_features.shape)

      for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(numpyPATH+path_of_feature, bf.numpy())



















