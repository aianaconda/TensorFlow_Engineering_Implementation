# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
model = __import__("5-2  model")
MyNASNetModel = model.MyNASNetModel

import sys                                      
nets_path = r'slim'                             #加载环境变量
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')
    
from nets.nasnet import nasnet                 #导出nasnet
slim = tf.contrib.slim                         #slim
image_size = nasnet.build_nasnet_mobile.default_image_size  #获得图片输入尺寸 224

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

batch_size = 32
test_dir  = 'data/val'



def check_accuracy(sess):
    """
    测试模型准确率
    """
    sess.run(mymode.test_init_op)  #初始化测试数据集
    num_correct, num_samples = 0, 0 #定义正确个数 和 总个数
    i = 0
    while True:
        i+=1
        print('i',i)
        try:
            #计算correct_prediction 获取prediction、labels是否相同 
            correct_pred,accuracy,logits = sess.run([mymode.correct_prediction,mymode.accuracy,mymode.logits])
            #累加correct_pred
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
            print("accuracy",accuracy,logits)

        
        except tf.errors.OutOfRangeError:  #捕获异常，数据用完自动跳出
            print('over')
            break
    
    acc = float(num_correct) / num_samples #计算并返回准确率
    return acc 


def check_sex(imgdir,sess):
    img = Image.open(image_dir)                               #读入图片
    if "RGB"!=img.mode :                                      #检查图片格式
        img = img.convert("RGB") 

    img = np.asarray(img.resize((image_size,image_size)),     #图像预处理  
                          dtype=np.float32).reshape(1,image_size,image_size,3)
    img = 2 *( img / 255.0)-1.0 

#一批次数据    
#    tt = img
#    for nn in range(31):
#        tt= np.r_[tt,img]
#    print(np.shape(tt))
    
    prediction = sess.run(mymode.logits, {mymode.images: img})#传入nasnet输入端中
    print(prediction)
    
    pre = prediction.argmax()#返回张量中最大值的索引

    print(pre)
    
    if pre == 1: img_id = 'man'
    elif pre == 2: img_id = 'woman'
    else: img_id = 'None'
    plt.imshow( np.asarray((img[0]+1)*255/2,np.uint8 )  )
    plt.show()
    print(img_id,"--",image_dir)#返回类别
    return pre
    

mymode = MyNASNetModel()                 #初始化模型
mymode.build_model('test',test_dir )     #将模型定义载入图中

with tf.Session() as sess:  
    #载入模型
    mymode.load_cpk(mymode.global_step,sess,1,mymode.saver,mymode.save_path )

    #测试模型的准确性
    val_acc = check_accuracy(sess)
    print('Val accuracy: %f\n' % val_acc)

    #单张图片测试
    image_dir = 'tt2t.jpg'         #选取测试图片
    check_sex(image_dir,sess)
    
    image_dir = test_dir + '\\woman' + '\\000001.jpg'         #选取测试图片
    check_sex(image_dir,sess)
    
    image_dir = test_dir + '\\man' + '\\000003.jpg'         #选取测试图片
    check_sex(image_dir,sess)
