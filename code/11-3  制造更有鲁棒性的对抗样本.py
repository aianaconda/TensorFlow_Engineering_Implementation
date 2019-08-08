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


batchsize =4

def pnasnetfunrotate(input_imgs,reuse ):
    rotatedarr = []
    for i in range(batchsize):
        rotated = tf.contrib.image.rotate(input_imgs,
                                          tf.random_uniform((), minval=-np.pi/4, maxval=np.pi/4))
        rotatedarr.append(tf.reshape(rotated,[1,image_size,image_size,3]))
 
    inputarr = tf.concat(rotatedarr,axis = 0)
    print(inputarr.get_shape())

    
    
    preprocessed = tf.subtract(tf.multiply(inputarr, 2.0), 1.0)#2 *( input_imgs / 255.0)-1.0  

    arg_scope = pnasnet.pnasnet_large_arg_scope() #获得模型命名空间

    
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.conv2d,
                             slim.batch_norm, slim.fully_connected,
                             slim.separable_conv2d],reuse=reuse):

            rotated_logits, end_points = pnasnet.build_pnasnet_large(preprocessed,num_classes = 1001, is_training=False)   
            prob = end_points['Predictions']  
    return rotated_logits, prob





input_imgs = tf.Variable(tf.zeros((image_size, image_size, 3))) 
#                      
rotated_logits, probs = pnasnetfunrotate(input_imgs,reuse=False)    
checkpoint_file = r'pnasnet-5_large_2017_12_13\model.ckpt'   #定义模型路径
variables_to_restore = slim.get_variables_to_restore()#(exclude=exclude)
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore,ignore_missing_vars=True)

sess = tf.InteractiveSession() #建立会话
init_fn(sess)                         #载入模型
              
img_path = './dog.jpg'
img = image.load_img(img_path, target_size=(image_size, image_size))
img = (np.asarray(img) / 255.0).astype(np.float32)

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

p = sess.run(probs, feed_dict={input_imgs: img})[0]
showresult(img,p)


def floatarr_to_img(floatarr):
    floatarr=np.asarray(floatarr*255)
    floatarr[floatarr>255]=255
    floatarr[floatarr<0]=0
    return floatarr.astype(np.uint8)
 
#开始攻击模型 
x = tf.placeholder(tf.float32, (image_size, image_size, 3)) #
assign_op = tf.assign(input_imgs, x)  #到目前为止input_imgs还没有初始化，必须调用input_imgs.initializer或tf.assign为其赋值
sess.run( assign_op, feed_dict={x: img})

below = input_imgs - 8.0/255.0  #定义图片的变化范围
above = input_imgs + 8.0/255.0

belowv,abovev = sess.run( [below,above])

plt.imshow(floatarr_to_img(belowv))#人眼验证
plt.show()
plt.imshow(floatarr_to_img(abovev))
plt.show()




label_target = 880
label =tf.constant(label_target)
labels = tf.tile([label],[batchsize])
print(labels.get_shape())

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=rotated_logits, labels=labels)  )

learning_rate=2e-1
optim_step_rotated = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, var_list=[input_imgs])

projected = tf.clip_by_value(tf.clip_by_value(input_imgs, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(input_imgs, projected)



demo_steps = 400

for i in range(demo_steps):
    _, loss_value = sess.run( [optim_step_rotated, loss])
    sess.run(project_step)
    if (i+1) % 10 == 0:
        print('step %d, loss=%g' % (i+1, loss_value))
        if loss_value<0.02:    #提前结束
            break
   
adv = input_imgs.eval() #获取图片

p = sess.run(probs)[0]
showresult(floatarr_to_img(adv),p)
plt.imsave('dog880rotated.jpg',floatarr_to_img(adv))

sess.close()









