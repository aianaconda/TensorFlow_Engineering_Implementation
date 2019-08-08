# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import sys                                      
nets_path = r'slim'                             #加载环境变量
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')
from nets.nasnet import nasnet                 #导出nasnet
slim = tf.contrib.slim                         #slim
image_size = nasnet.build_nasnet_mobile.default_image_size  #获得图片输入尺寸 224
from preprocessing import preprocessing_factory#图像处理


import os
def list_images(directory):
    """
    获取所有directory中的所有图片和标签
    """

    #返回path指定的文件夹包含的文件或文件夹的名字的列表
    labels = os.listdir(directory)
    #对标签进行排序，以便训练和验证按照相同的顺序进行
    labels.sort()
    #创建文件标签列表
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            #转换字符串中所有大写字符为小写再判断
            if 'jpg' in f.lower() or 'png' in f.lower():
                #加入列表
                files_and_labels.append((os.path.join(directory, label, f), label))
    #理解为解压 把数据路径和标签解压出来
    filenames, labels = zip(*files_and_labels)
    #转换为列表 分别储存数据路径和对应标签
    filenames = list(filenames)
    labels = list(labels)
    #列出分类总数 比如两类：['man', 'woman']
    unique_labels = list(set(labels))

    label_to_int = {}
    #循环列出数据和数据下标
    #给每个分类打上标签{'woman': 2, 'man': 1，none：0}
    for i, label in enumerate(sorted(unique_labels)):
        label_to_int[label] = i+1
    print(label,label_to_int[label])
    #把每个标签化为0 1 这种形式
    labels = [label_to_int[l] for l in labels]
    print(labels[:6],labels[-6:])
    return filenames, labels  #返回储存数据路径和对应转换后的标签


num_workers = 2  #定义并行处理数据的线程数量

#图像批量预处理
image_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile', is_training=True)
image_eval_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile', is_training=False)

def _parse_function(filename, label):  #定义图像解码函数
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)          
    return image, label

def training_preprocess(image, label):  #定义调整图像大小函数
    image = image_preprocessing_fn(image, image_size, image_size)
    return image, label

def val_preprocess(image, label):   #定义评估图像预处理函数
    image = image_eval_preprocessing_fn(image, image_size, image_size)
    return image, label

#创建带批次的数据集
def creat_batched_dataset(filenames, labels,batch_size,isTrain = True):
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.map(_parse_function, num_parallel_calls=num_workers)#对图像解码
        
    if isTrain == True:
        dataset = dataset.shuffle(buffer_size=len(filenames))#打乱数据顺序
        dataset = dataset.map(training_preprocess, num_parallel_calls=num_workers)#调整图像大小
    else:
        dataset = dataset.map(val_preprocess,num_parallel_calls=num_workers)#调整图像大小
        
    return dataset.batch(batch_size)   #返回批次数据

#根据目录返回数据集
def creat_dataset_fromdir(directory,batch_size,isTrain = True):
    filenames, labels = list_images(directory)
    num_classes = len(set(labels))
    print("num_classes",num_classes)
    dataset = creat_batched_dataset(filenames, labels,batch_size,isTrain)
    return dataset,num_classes