# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import os
import tensorflow as tf
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
generator = __import__("8-14  generator")
BatchGenerator = generator.BatchGenerator
box = __import__("8-15  box")
draw_boxes = box.draw_boxes
yolov3 = __import__("8-18  yolov3")
Yolonet = yolov3.Yolonet
yololoss = __import__("8-20  yololoss")
loss_fn = yololoss.loss_fn

tf.enable_eager_execution()

PROJECT_ROOT = os.path.dirname(__file__)#获取当前目录
print(PROJECT_ROOT)

#定义coco锚点候选框
COCO_ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
#定义预训练模型路径
YOLOV3_WEIGHTS = os.path.join(PROJECT_ROOT, "yolov3.weights")
#定义分类
LABELS = ['0',"1", "2", "3",'4','5','6','7','8', "9"]

#定义样本路径
ann_dir = os.path.join(PROJECT_ROOT,  "data", "ann", "*.xml")
img_dir = os.path.join(PROJECT_ROOT,  "data", "img")

train_ann_fnames = glob.glob(ann_dir)#获取该路径下的xml文件
   
imgsize =416
batch_size =2
#制作数据集
generator = BatchGenerator(train_ann_fnames,img_dir,
                           net_size=imgsize,
                           anchors=COCO_ANCHORS,
                             batch_size=2,
                             labels=LABELS,
                             jitter = False)#随机变化尺寸。数据增强

#定义训练参数
learning_rate = 1e-4  #定义学习率
num_epoches =85       #定义迭代次数
save_dir = "./model"  #定义模型路径

#循环整个数据集，进行loss值验证
def _loop_validation(model, generator):
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in range(n_steps): #按批次循环获取数据，并计算loss
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        xs=tf.convert_to_tensor(xs)
        yolo_1=tf.convert_to_tensor(yolo_1)
        yolo_2=tf.convert_to_tensor(yolo_2)
        yolo_3=tf.convert_to_tensor(yolo_3)        
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs )
        loss_value += loss_fn(ys, ys_,anchors=COCO_ANCHORS,
            image_size=[imgsize, imgsize] )
    loss_value /= generator.steps_per_epoch
    return loss_value

#循环整个数据集，进行模型训练
def _loop_train(model,optimizer, generator,grad):
    # one epoch
    n_steps = generator.steps_per_epoch
    for _ in tqdm(range(n_steps)):#按批次循环获取数据，并计算训练
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        xs=tf.convert_to_tensor(xs)
        yolo_1=tf.convert_to_tensor(yolo_1)
        yolo_2=tf.convert_to_tensor(yolo_2)
        yolo_3=tf.convert_to_tensor(yolo_3)
        ys = [yolo_1, yolo_2, yolo_3]
        optimizer.apply_gradients(grad(model,xs, ys))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_fname = os.path.join(save_dir, "weights")

yolo_v3 = Yolonet(n_classes=len(LABELS))#实例化yolo模型类对象
yolo_v3.load_darknet_params(YOLOV3_WEIGHTS, skip_detect_layer=True)#加载预训练模型

#定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

#定义函数计算loss
def _grad_fn(yolo_v3, images_tensor, list_y_trues):
    logits = yolo_v3(images_tensor)   
    loss = loss_fn(list_y_trues, logits,anchors=COCO_ANCHORS,
            image_size=[imgsize, imgsize])
    return loss

grad = tfe.implicit_gradients(_grad_fn)#获得计算梯度的函数

history = []
for i in range(num_epoches):
    _loop_train( yolo_v3,optimizer, generator,grad)#训练

    loss_value = _loop_validation(yolo_v3, generator)#验证
    print("{}-th loss = {}".format(i, loss_value))

    #收集loss
    history.append(loss_value)
    if loss_value == min(history):#只有loss创新低时再保存模型
        print("    update weight {}".format(loss_value))
        yolo_v3.save_weights("{}.h5".format(save_fname))
################################################################
#使用模型

IMAGE_FOLDER = os.path.join(PROJECT_ROOT,  "data", "test","*.png")
img_fnames = glob.glob(IMAGE_FOLDER)

imgs = []   #存放图片
for fname in img_fnames:#读取图片
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

yolo_v3.load_weights(save_fname+".h5")#将训练好的模型载入
import numpy as np
for img in imgs:  #依次传入模型
    boxes, labels, probs = yolo_v3.detect(img, COCO_ANCHORS,imgsize)
    print(boxes, labels, probs)
    image = draw_boxes(img, boxes, labels, probs, class_labels=LABELS, desired_size=400)
    image = np.asarray(image,dtype= np.uint8)
    plt.imshow(image)
    plt.show()



