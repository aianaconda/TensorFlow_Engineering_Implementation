# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import sys                                         #初始化环境变量
nets_path = r'slim'
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')

import tensorflow as tf                           #引入头文件
from nets.nasnet import pnasnet
import numpy as np
from tensorflow.python.keras.preprocessing import image

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签  
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 15

slim = tf.contrib.slim
arg_scope = tf.contrib.framework.arg_scope

tf.reset_default_graph()                       
image_size = pnasnet.build_pnasnet_large.default_image_size   #获得图片输入尺寸
LANG = 'ch'            #使用中文标签

if LANG=='ch':
    def getone(onestr):
        return onestr.replace(',',' ').replace('\n','')
    
    with open('中文标签.csv','r+') as f: 		#打开文件				
        labelnames =list( map(getone,list(f))  )
        print(len(labelnames),type(labelnames),labelnames[:5]) #显示输出中文标签    
else: 
    from datasets import imagenet
    labelnames = imagenet.create_readable_names_for_imagenet_labels() #获得数据集标签
    print(len(labelnames),labelnames[:5])                                     #显示输出标签


def pnasnetfun(input_imgs,reuse ):
    preprocessed = tf.subtract(tf.multiply(tf.expand_dims(input_imgs, 0), 2.0), 1.0)#2 *( input_imgs / 255.0)-1.0  

    arg_scope = pnasnet.pnasnet_large_arg_scope() #获得模型命名空间
    
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.conv2d,
                             slim.batch_norm, slim.fully_connected,
                             slim.separable_conv2d],reuse=reuse):

            logits, end_points = pnasnet.build_pnasnet_large(preprocessed,num_classes = 1001, is_training=False)   
            prob = end_points['Predictions']  
    return logits, prob



input_imgs = tf.Variable(tf.zeros((image_size, image_size, 3)))                        
logits, probs = pnasnetfun(input_imgs,reuse=False)    
checkpoint_file = r'pnasnet-5_large_2017_12_13\model.ckpt'   #定义模型路径
variables_to_restore = slim.get_variables_to_restore()#(exclude=exclude)
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore,ignore_missing_vars=True)

sess = tf.InteractiveSession() #建立会话
init_fn(sess)                         #载入模型

def showresult(img,p):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)

    ax1.axis('off')
    ax1.imshow(img)
    fig.sca(ax1)
    
    top10 = list((-p).argsort()[:10])
    lab= [labelnames[i][:15] for i in top10]
    topprobs = p[top10]
    print(list(zip(top10,lab,topprobs)))
    
    barlist = ax2.bar(range(10), topprobs)

    barlist[0].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10), lab, rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()




img_path = './dog880.jpg'
imgtest = image.load_img(img_path, target_size=(image_size, image_size))
imgtest = (np.asarray(imgtest) / 255.0).astype(np.float32)


ex_angle = np.pi/8
angle = tf.placeholder(tf.float32, ())
rotated_image = tf.contrib.image.rotate(input_imgs, angle)
rotated_example = rotated_image.eval(feed_dict={input_imgs: imgtest, angle: ex_angle})
p = sess.run(probs, feed_dict={input_imgs: rotated_example})[0]
showresult(rotated_example,p)   #仍然能识别




#两个攻击样本在模型下的对比
img_path = './dog880rotated.jpg'
imgtestrotated = image.load_img(img_path, target_size=(image_size, image_size))
imgtestrotated = (np.asarray(imgtestrotated) / 255.0).astype(np.float32)

thetas = np.linspace(-np.pi/4, np.pi/4, 301)
label_target = 880
p_naive = []
p_robust = []
for theta in thetas:
    rotated = rotated_image.eval(feed_dict={input_imgs: imgtestrotated, angle: theta})
    p_robust.append(probs.eval(feed_dict={input_imgs: rotated})[0][label_target])
    
    rotated = rotated_image.eval(feed_dict={input_imgs: imgtest, angle: theta})
    p_naive.append(probs.eval(feed_dict={input_imgs: rotated})[0][label_target])

robust_line, = plt.plot(thetas, p_robust, color='b', linewidth=2, label='支持旋转的攻击样本')
naive_line, = plt.plot(thetas, p_naive, color='r', linewidth=2, label='不支持旋转攻击样本')
plt.ylim([0, 1.05])
plt.xlabel('旋转角度')
plt.ylabel('880类别的概率')
plt.legend(handles=[robust_line, naive_line], loc='lower right')
plt.show()

sess.close()
