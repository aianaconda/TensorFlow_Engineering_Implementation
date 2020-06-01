"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def load_sample(sample_dir):
    '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
    print ('loading sample  dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#递归遍历文件夹
        for filename in filenames:                            #遍历所有文件名
            #print(dirnames)
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #添加文件名
            labelsnames.append( dirpath.split('\\')[-1] )#添加文件名对应的标签

    lab= list(sorted(set(labelsnames)))  #生成标签名称列表
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #生成字典

    labels = [labdict[i] for i in labelsnames]
    return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)


data_dir = 'mnist_digits_images\\'  #定义文件路径

(image,label),labelsnames = load_sample(data_dir)   #载入文件名称与标签
print(len(image),image[:2],len(label),label[:2])#输出load_sample返回的数据结果
print(labelsnames[ label[:2] ],labelsnames)#输出load_sample返回的标签字符串


def get_batches(image,label,input_w,input_h,channels,batch_size):

    queue = tf.train.slice_input_producer([image,label])  #使用tf.train.slice_input_producer实现一个输入的队列
    label = queue[1]                                        #从输入队列里读取标签

    image_c = tf.read_file(queue[0])                        #从输入队列里读取image路径

    image = tf.image.decode_bmp(image_c,channels)           #按照路径读取图片

    image = tf.image.resize_image_with_crop_or_pad(image,input_w,input_h) #修改图片大小


    image = tf.image.per_image_standardization(image) #图像标准化处理，(x - mean) / adjusted_stddev

    image_batch,label_batch = tf.train.batch([image,label],#调用tf.train.batch函数生成批次数据
               batch_size = batch_size,
               num_threads = 64)

    images_batch = tf.cast(image_batch,tf.float32)   #将数据类型转换为float32

    labels_batch = tf.reshape(label_batch,[batch_size])#修改标签的形状shape
    return images_batch,labels_batch


batch_size = 16
image_batches,label_batches = get_batches(image,label,28,28,1,batch_size)



def showresult(subplot,title,thisimg):          #显示单个图片
    p =plt.subplot(subplot)
    p.axis('off')
    #p.imshow(np.asarray(thisimg[0], dtype='uint8'))
    p.imshow(np.reshape(thisimg, (28, 28)))
    p.set_title(title)

def showimg(index,label,img,ntop):   #显示
    plt.figure(figsize=(20,10))     #定义显示图片的宽、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  #初始化

    coord = tf.train.Coordinator()          #开启列队
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    try:
        for step in np.arange(10):
            if coord.should_stop():
                break
            images,label = sess.run([image_batches,label_batches]) #注入数据

            showimg(step,label,images,batch_size)       #显示图片
            print(label)                                 #打印数据

    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()

    coord.join(threads)                             #关闭列队

