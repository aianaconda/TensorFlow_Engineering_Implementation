"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import os
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

tf.compat.v1.disable_v2_behavior()

def load_sample(sample_dir,shuffleflag = True):
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
    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)



directory='man_woman\\'                                                     #定义样本路径
(filenames,labels),_ = load_sample(directory,shuffleflag=False)   #载入文件名称与标签


def makeTFRec(filenames,labels): #定义函数生成TFRecord
    writer= tf.io.TFRecordWriter("mydata.tfrecords") #通过tf.io.TFRecordWriter 写入到TFRecords文件
    for i in tqdm( range(0,len(labels) ) ):
        img=Image.open(filenames[i])
        img = img.resize((256, 256))
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
                #存放图片的标签label
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                #存放具体的图片
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example对象对label和image数据进行封装

        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()  #数据集制作完成

makeTFRec(filenames,labels)

################将tf数据集转化为图片##########################
def read_and_decode(filenames,flag = 'train',batch_size = 3):
    #根据文件名生成一个队列
    if flag == 'train':
        filename_queue = tf.compat.v1.train.string_input_producer(filenames)#默认已经是shuffle并且循环读取
    else:
        filename_queue = tf.compat.v1.train.string_input_producer(filenames,num_epochs = 1,shuffle = False)

    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.io.parse_single_example(serialized=serialized_example, #取出包含image和label的feature对象
                                       features={
                                           'label': tf.io.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.io.FixedLenFeature([], tf.string),
                                       })

    #tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.io.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [256,256,3])
    #
    label = tf.cast(features['label'], tf.int32)

    if flag == 'train':
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5     #归一化
        img_batch, label_batch = tf.compat.v1.train.batch([image, label],   #还可以使用tf.train.shuffle_batch进行乱序批次
                                                batch_size=batch_size, capacity=20)
#        img_batch, label_batch = tf.train.shuffle_batch([image, label],
#                                        batch_size=batch_size, capacity=20,
#                                        min_after_dequeue=10)
        return img_batch, label_batch

    return image, label

#############################################################
TFRecordfilenames = ["mydata.tfrecords"]
image, label =read_and_decode(TFRecordfilenames,flag='test')  #以测试的方式打开数据集


saveimgpath = 'show\\'    #定义保存图片路径
if tf.io.gfile.exists(saveimgpath):  #如果存在saveimgpath，将其删除
    tf.io.gfile.rmtree(saveimgpath)  #也可以使用shutil.rmtree(saveimgpath)
tf.io.gfile.makedirs(saveimgpath)    #创建saveimgpath路径

#开始一个会话读取数据
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.local_variables_initializer())   #初始化本地变量，没有这句会报错
    #启动多线程
    coord=tf.train.Coordinator()
    threads= tf.compat.v1.train.start_queue_runners(coord=coord)
    myset = set([])

    try:
        i = 0
        while True:
            example, examplelab = sess.run([image,label])#在会话中取出image和label
            examplelab = str(examplelab)
            if examplelab not in myset:
                myset.add(examplelab)
                tf.io.gfile.makedirs(saveimgpath+examplelab)
                print(saveimgpath+examplelab,i)
            img=Image.fromarray(example, 'RGB')#转换Image格式
            img.save(saveimgpath+examplelab+'/'+str(i)+'_Label_'+'.jpg')#存下图片
            print( i)
            i = i+1
    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        print("stop()")
#############################################################
#训练方式
image, label =read_and_decode(TFRecordfilenames)  #以训练的方式打开数据集
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.local_variables_initializer())   #初始化本地变量，没有这句会报错
    #启动多线程
    coord=tf.train.Coordinator()
    threads= tf.compat.v1.train.start_queue_runners(coord=coord)
    myset = set([])
    try:
        for i in range(5):
            example, examplelab = sess.run([image,label])#在会话中取出image和label

            dirtrain = saveimgpath+"train_"+str(i)
            print(dirtrain,examplelab)
            tf.io.gfile.makedirs(dirtrain)
            for lab in range(len(examplelab)):
                print(lab)
                img=Image.fromarray(example[lab], 'RGB')#这里Image是之前提到的
                img.save(dirtrain+'/'+str(lab)+'_Label_'+str(examplelab[lab])+'.jpg')#存下图片

    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        print("stop()")