# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

annFile='./cocos2014/annotations_trainval2014/annotations/instances_train2014.json'
coco=COCO(annFile)#加载注解的json数据

cats = coco.loadCats(coco.getCatIds())#提取分类信息
print(cats,len(cats))#80个分类
nmcats=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nmcats)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
print("supercategory len",len(nms))#12个超级分类

# 分类并不连续 ，例如：没有26.第一个是1，  最后一个是90.
catIds = coco.getCatIds(catNms=nmcats)
print(catIds)  

#根据类名获得对应的图片列表
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )  
print(catIds,len(imgIds),imgIds[:5])

#从指定列表中取一张图片
index = imgIds[np.random.randint(0,len(imgIds))]
print(index)
img = coco.loadImgs(index)[0]#index可以是数组。会返回多个图片
print(img)
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()


#获得标注的分割信息，并叠加到原图显示出来
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)#iscrowd代表是否是一群
anns = coco.loadAnns(annIds)#一条标注ID对应的信息（segmentation（分割）、bbox（框）、category_id（类别））
print(annIds,anns)
coco.showAnns(anns)#将分割的信息叠加到图像上

#加载关键点json
annFile = './annotations_trainval2014/annotations/person_keypoints_train2014.json'
coco_kps=COCO(annFile)
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)#超级类person的每条标注，包括了关键点 及segmentation和bbox、category_id
print(annIds,anns)
coco_kps.showAnns(anns)

#加载图片描述json
annFile = './annotations_trainval2014/annotations/captions_train2014.json'
coco_caps=COCO(annFile)
annIds = coco_caps.getAnnIds(imgIds=img['id']);#每一个图片id,对应多条描述
anns = coco_caps.loadAnns(annIds)#跟据描述id，载入每条描述
print(annIds,anns)#每条描述包括id 图片id 和一句话
coco_caps.showAnns(anns)


