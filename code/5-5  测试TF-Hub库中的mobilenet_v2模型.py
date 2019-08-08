# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


with open('中文标签.csv','r+') as f: 		#打开文件
    labels =list( map(lambda x:x.replace(',',' '),list(f))  )
    print(len(labels),type(labels),labels[:5]) #显示输出中文标签

sample_images = ['hy.jpg', 'ps.jpg','72.jpg']               #定义待测试图片路径

#加载分类模型
module_spec = hub.load_module_spec("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2")

#module_spec = hub.load_module_spec("mobilenet_v2_100_224")

#module_spec = hub.load_module_spec(r"C:\Users\ljh\AppData\Local\Temp\tfhub_modules\bb6444e8248f8c581b7a320d5ff53061e4506c19")
height, width = hub.get_expected_image_size(module_spec)#获得模型的输入图片尺寸

input_imgs = tf.placeholder(tf.float32, [None, height,width,3]) #定义占位符[batch_size, height, width, 3].
images = 2 *( input_imgs / 255.0)-1.0                         #归一化图片


module = hub.Module(module_spec)

logits = module(images)   # 输出的形状为 [batch_size, num_classes].
#也可以使用如下代码（以签名的方式）
#  outputs = module(dict(images=images), signature="image_classification", as_dict=True)
#  logits = outputs["default"]

y = tf.argmax(logits,axis = 1)                          #获得结果的输出节点
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    def preimg(img):                                    #定义图片预处理函数
        return np.asarray(img.resize((height, width)),
                          dtype=np.float32).reshape(height, width,3)

    #获得原始图片与预处理图片
    batchImg = [ preimg( Image.open(imgfilename) ) for imgfilename in sample_images ]
    orgImg = [  Image.open(imgfilename)  for imgfilename in sample_images ]

    yv,img_norm = sess.run([y,images], feed_dict={input_imgs: batchImg})    #输入到模型

    print(yv,np.shape(yv))                                  #显示输出结果
    def showresult(yy,img_norm,img_org):                    #定义显示图片函数
        plt.figure()
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)
        p1.imshow(img_org)# 显示图片
        p1.axis('off')
        p1.set_title("organization image")

        p2.imshow((img_norm * 255).astype(np.uint8))# 显示图片
        p2.axis('off')
        p2.set_title("input image")

        plt.show()

        print(yy,labels[yy])

    for yy,img1,img2 in zip(yv,batchImg,orgImg):            #显示每条结果及图片
        showresult(yy,img1,img2)





