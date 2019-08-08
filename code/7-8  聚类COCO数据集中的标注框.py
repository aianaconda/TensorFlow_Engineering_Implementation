# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

usecoco = 1  #实例的演示方式。设为1，表示使用coco数据集

def convert_coco_bbox(size, box):
    """
    Introduction
    ------------
        计算box的长宽和原始图像的长宽比值
    Parameters
    ----------
        size: 原始图像大小
        box: 标注box的信息
    Returns
        x, y, w, h 标注box和原始图像的比值
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def load_cocoDataset(annfile):
    """
    Introduction
    ------------
        读取coco数据集的标注信息
    Parameters
    ----------
        datasets: 数据集名字列表
    """
    from pycocotools.coco import COCO
    #from six.moves import xrange
    data = []
    coco = COCO(annfile)
    cats = coco.loadCats(coco.getCatIds())
    coco.loadImgs()
    base_classes = {cat['id'] : cat['name'] for cat in cats}
    imgId_catIds = [coco.getImgIds(catIds = cat_ids) for cat_ids in base_classes.keys()]
    image_ids = [img_id for img_cat_id in imgId_catIds for img_id in img_cat_id ]
    for image_id in image_ids:
        annIds = coco.getAnnIds(imgIds = image_id)
        anns = coco.loadAnns(annIds)
        img = coco.loadImgs(image_id)[0]
        image_width = img['width']
        image_height = img['height']

        for ann in anns:
            box = ann['bbox']
            bb = convert_coco_bbox((image_width, image_height), box)
            data.append(bb[2:])
    return np.array(data)




if usecoco == 1:
    dataFile = r"E:\Mask_RCNN-master\cocos2014\annotations\instances_train2014.json"
    points = load_cocoDataset(dataFile)
else: 
    num_points = 100
    dimensions = 2
    points = np.random.uniform(0, 1000, [num_points, dimensions])


num_clusters = 5
config=tf.estimator.RunConfig(model_dir='./kmeansmodel',save_checkpoints_steps=100)

kmeans = tf.contrib.factorization.KMeansClustering(config= config,
    num_clusters=num_clusters, use_mini_batch=False,relative_tolerance=0.01)

#训练部分
def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=300)
kmeans.train(input_fn)
print("训练结束，score(cost) = {}".format(kmeans.score(input_fn)))

anchors = kmeans.cluster_centers()

box_w = points[:1000, 0]
box_h = points[:1000, 1]
#plt.scatter(box_h, box_w, c = 'r')
#
#print(len(anchors))
#anchors = np.asarray(anchors)
#print((anchors[:,0]))
#plt.scatter(anchors[:,0], anchors[:, 1], c = 'b')
#plt.show() 

#聚类结果
def show_input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points[:1000], dtype=tf.float32), num_epochs=1)
cluster_indices =list( kmeans.predict_cluster_index(show_input_fn) )


plt.scatter(box_h, box_w, c=cluster_indices)
plt.colorbar()
plt.scatter(anchors[:,0], anchors[:, 1], s=800,c='r',marker='x')
plt.show()

if usecoco == 1:
    trueanchors = []
    for cluster in anchors:
        trueanchors.append([round(cluster[0] * 416), round(cluster[1] * 416)])
    print("在416*416上面，所聚类的锚点候选框为：",trueanchors)
 

distance = list(kmeans.transform(show_input_fn))#获得每个坐标离中心点的距离  
predict = list(kmeans.predict(show_input_fn) )#对每个点进行预测
print(distance[0],predict[0]) #显示内容

#取出第一个类的数据。并按照中心点远近排序
firstclassdistance= np.array([  p['all_distances'][0]  for p in predict if p['cluster_index']==0 ])
dataindexsort= np.argsort(firstclassdistance)
print(len(dataindexsort),dataindexsort[:10],firstclassdistance[dataindexsort[:10]])


  
  
  
  
  
  