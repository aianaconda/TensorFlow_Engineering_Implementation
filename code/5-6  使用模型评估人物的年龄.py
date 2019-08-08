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
from sklearn.utils import shuffle
import os

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


#载入标签
data_dir = 'IMBD-WIKI\\'  #定义文件路径
_,labels = load_sample(data_dir,False)   #载入文件名称与标签
print(labels)#输出load_sample返回的标签字符串


sample_images = ['22.jpg', 'tt2t.jpg']               #定义待测试图片路径



tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()
#分类模型
thissavedir= 'tmp'
PATH_TO_CKPT = thissavedir +'/output_graph.pb'
od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
#    print(od_graph_def)
    tf.import_graph_def(od_graph_def, name='')

fenlei_graph = tf.get_default_graph()


#print(fenlei_graph.get_operations())

height,width = 224,224

with tf.Session(graph=fenlei_graph) as sess:
    result = fenlei_graph.get_tensor_by_name('final_result:0')
    input_imgs = fenlei_graph.get_tensor_by_name('Placeholder:0')
    y = tf.argmax(result,axis = 1)


    def preimg(img):                                    #定义图片预处理函数
        reimg = np.asarray(img.resize((height, width)),
                          dtype=np.float32).reshape(height, width,3)
        normimg = 2 *( reimg / 255.0)-1.0
        return normimg

    #获得原始图片与预处理图片
    batchImg = [ preimg( Image.open(imgfilename) ) for imgfilename in sample_images ]
    orgImg = [  Image.open(imgfilename)  for imgfilename in sample_images ]

    yv = sess.run(y, feed_dict={input_imgs: batchImg})
    print(yv)

    print(yv,np.shape(yv))                                  #显示输出结果
    def showresult(yy,img_norm,img_org):                    #定义显示图片函数
        plt.figure()
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)
        p1.imshow(img_org)# 显示图片
        p1.axis('off')
        p1.set_title("organization image")

        img = ((img_norm+1)/2)*255
        p2.imshow(  np.asarray(img,np.uint8)      )# 显示图片
        p2.axis('off')
        p2.set_title("input image")

        plt.show()

        print("索引：",yy,",","年纪：",labels[yy])

    for yy,img1,img2 in zip(yv,batchImg,orgImg):            #显示每条结果及图片
        showresult(yy,img1,img2)




