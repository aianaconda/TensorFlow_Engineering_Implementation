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
from pycocotools.coco import COCO
import skimage.io as io

mask_rcnn_model = __import__("8-24  mask_rcnn_model")
MaskRCNN = mask_rcnn_model.MaskRCNN
utils = __import__("8-25  mask_rcnn_utils")
visualize = __import__("8-26  mask_rcnn_visualize")

################################加载数据集
annFile='./cocos2014/annotations_trainval2014/annotations/instances_train2014.json'
coco=COCO(annFile)#加载注解的json数据

class_ids = sorted(coco.getCatIds())#获得分类id
class_info = coco.loadCats(coco.getCatIds())#提取分类信息
class_name=[n["name"] for n in class_info]

class_ids.insert(0,0)
class_name.insert(0,"BG")

print(class_ids)#所有的类索引
print(class_name)#所有的类名


#################################################载入模型

BATCH_SIZE =1#批次

MODEL_DIR = "./log"
#可以让gpu进行训练，cpu进行检测。但是实际中，如果gpu配置低，直接gpu有可能会内核死掉。
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
# Create model in inference mode
with tf.device(DEVICE):
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, num_class=len(class_ids),batch_size = BATCH_SIZE)#加完背景后81个类

#模型权重文件路径
weights_path = "./mask_rcnn_coco.h5"

#载入权重文件
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# Show stats of all trainable weights    
utils.html_weight_stats(model)#显示权重
######################################################

################################################获取一个文件
#根据类名获得对应的图片列表
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )  
print(catIds,len(imgIds),imgIds[:5])

#从指定列表中取一张图片
index = imgIds[np.random.randint(0,len(imgIds))]
print(index)
img = coco.loadImgs(index)[0]#index可以是数组。会返回多个图片
print(img)
image = io.imread(img['coco_url'])#从网络加载一个图片
plt.axis('off')
plt.imshow(image)
plt.show()

####################################################################
#骨干网结果
  
ResNetFeatures = utils.run_graph(model,[image], [
    ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
    ("res5c_out",          model.keras_model.get_layer("res5c_out").output),  # for resnet100
],BATCH_SIZE)
    
# Backbone feature map(1, 64, 64, 1024)
visualize.display_images(np.transpose(ResNetFeatures["res4w_out"][0,:,:,:4], [2, 0, 1]))
visualize.display_images(np.transpose(ResNetFeatures["res5c_out"][0,:,:,:4], [2, 0, 1]))

##################################################
#FPN结果
roi_align_mask = utils.run_graph(model,[image], [
    ("fpn_p2",          model.keras_model.get_layer("fpn_p2").output),  #
    ("fpn_p3",          model.keras_model.get_layer("fpn_p3").output),  #
    ("fpn_p4",          model.keras_model.get_layer("fpn_p4").output),  #
    ("fpn_p5",          model.keras_model.get_layer("fpn_p5").output),  #   
    ("fpn_p6",          model.keras_model.get_layer("fpn_p6").output),  #    
],BATCH_SIZE)


###################################################
#第一步 RPN网络
    
# Run RPN sub-graph
pillar = model.keras_model.get_layer("ROI").output  #获得ROI节点，即 ProposalLayer层

rpn = utils.run_graph(model,[image], [
    ("rpn_class", model.keras_model.get_layer("rpn_class").output),#(1, 261888, 2) 
    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
    ("post_nms_anchor_ix", model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0") ),#shape: (1000,)
    ("proposals", model.keras_model.get_layer("ROI").output),
],BATCH_SIZE)

print(rpn['rpn_class'][0,:3])#将rpn网络的前三个元素打印出来
print(rpn['pre_nms_anchors'][0,:3])#将rpn网络的前三个元素打印出来    
print(model.anchors[:3])#将rpn网络的前三个元素打印出来    
    
def get_ax(rows=1, cols=1, size=16):#设置显示的图片位置及大小
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
#将前50个分值高的锚点示出来
limit = 50
h, w = mask_rcnn_model.IMAGE_DIM,mask_rcnn_model.IMAGE_DIM;
pre_nms_anchors = rpn['pre_nms_anchors'][0, :limit] * np.array([h, w, h, w])
print(image.shape)
image2, window, scale, padding, _ = utils.resize_image( image, 
                                    min_dim=mask_rcnn_model.IMAGE_MIN_DIM, 
                                    max_dim=mask_rcnn_model.IMAGE_MAX_DIM,
                                    mode=mask_rcnn_model.IMAGE_RESIZE_MODE)
print(image2.shape)
visualize.draw_boxes(image2, boxes=pre_nms_anchors, ax=get_ax())


#rpn_class (1, 261888, 2)  背景和前景。softmax后的值。1代表正值， 由大到小的
sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
visualize.draw_boxes(image2, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())



#随意取50个带nms之前的数据，和调整后的。  还有规整完调整后的  (1, 6000, 4)
ax = get_ax(1, 2)
pre_nms_anchors = rpn['pre_nms_anchors'][0, :limit] * np.array([h, w, h, w])
refined_anchors = rpn['refined_anchors'][0, :limit] * np.array([h, w, h, w])
refined_anchors_clipped = rpn['refined_anchors_clipped'][0, :limit] * np.array([h, w, h, w])
#取50个在nms之前的数据，边框调整后和边框剪辑后的
visualize.draw_boxes(image2, boxes=pre_nms_anchors,refined_boxes=refined_anchors, ax=ax[0])
visualize.draw_boxes(image2, refined_boxes=refined_anchors_clipped, ax=ax[1])#边框剪辑后的
######################################################

#用nms之后(1000,)为具体的索引值
post_nms_anchor_ix = rpn['post_nms_anchor_ix'][ :limit]
refined_anchors_clipped = rpn["refined_anchors_clipped"][0, post_nms_anchor_ix] * np.array([h, w, h, w])
visualize.draw_boxes(image2, refined_boxes=refined_anchors_clipped, ax=get_ax())


# Convert back to image coordinates for display(1, 1000, 4) 
proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
visualize.draw_boxes(image2, refined_boxes=proposals, ax=get_ax())

########################################################

######################################################
#第二步
    
roi_align_classifierlar = model.keras_model.get_layer("roi_align_classifier").output  #获得ROI节点，即 ProposalLayer层

roi_align_classifier = utils.run_graph(model,[image], [
    ("roi_align_classifierlar", model.keras_model.get_layer("roi_align_classifier").output),#(1, 261888, 2) 
    ("ix", model.ancestor(roi_align_classifierlar, "roi_align_classifier/ix:0")),
    ("level_boxes", model.ancestor(roi_align_classifierlar, "roi_align_classifier/level_boxes:0")),
    ("box_indices", model.ancestor(roi_align_classifierlar, "roi_align_classifier/Cast_2:0")),

 ],BATCH_SIZE)
    
print(roi_align_classifier["ix"][:5])    #(828, 2) 
print(roi_align_classifier["level_boxes"][:5])  #(828, 4)      
print(roi_align_classifier["box_indices"][:5])  #(828, 4) 



#分类器结果
fpn_classifier = utils.run_graph(model,[image], [
    ("probs", model.keras_model.get_layer("mrcnn_class").output),#shape: (1, 1000, 81)
    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),#(1, 1000, 81, 4) 
],BATCH_SIZE)
#相对于reshape后的框，所以要使用image2
proposals=utils.denorm_boxes(rpn["proposals"][0], image2.shape[:2])#(1000, 4)

#81类中的最大索引--代表class id(索引就是分类)
roi_class_ids = np.argmax(fpn_classifier["probs"][0], axis=1)#(1000,)
print(roi_class_ids.shape,roi_class_ids[:20])
roi_class_names = np.array(class_name)[roi_class_ids]#根据索引把名字取出来
print(roi_class_names[:20])
#去重，类别个数
print(list(zip(*np.unique(roi_class_names, return_counts=True))))

roi_positive_ixs = np.where(roi_class_ids > 0)[0]#不是背景的类索引
print("{}中有{}个前景实例\n{}".format(len(proposals),len(roi_positive_ixs),roi_positive_ixs))

#根据索引将最大的那个值取出来。当作分数
roi_scores = np.max(fpn_classifier["probs"][0],axis=1)
print(roi_scores.shape,roi_scores[:20])

############################
#边框可视化
limit = 50
ax = get_ax(1, 2)

ixs = np.random.randint(0, proposals.shape[0], limit)
captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]

visib= np.where(roi_class_ids[ixs] > 0, 2, 1)#前景统一设为2，背景设为1

visualize.draw_boxes(image2, boxes=proposals[ixs],  #原始的框放进去
                     visibilities=visib,#2突出显示.1一般显示
                     captions=captions, title="before fpn_classifier", ax=ax[0])

#把指定类索引的坐标提取出来
#取出每个框对应分类的坐标偏移。fpn_classifier["deltas"]形状为(1, 1000, 81, 4)
roi_bbox_specific = fpn_classifier["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
print("roi_bbox_specific", roi_bbox_specific)#( 1000,  4)

#根据偏移来调整ROI Shape: [N, (y1, x1, y2, x2)]
refined_proposals = utils.apply_box_deltas(
    proposals, roi_bbox_specific * mask_rcnn_model.BBOX_STD_DEV).astype(np.int32)
print("refined_proposals", refined_proposals)

limit =5
ids = np.random.randint(0, len(roi_positive_ixs), limit)  #取出5个不是背景的类

captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]

visualize.draw_boxes(image2, boxes=proposals[roi_positive_ixs][ids],
                     refined_boxes=refined_proposals[roi_positive_ixs][ids],
                     captions=captions, title="After fpn_classifier",ax=ax[1])


#如果将坐标按照图片image来变化，还需要如下的方法转成image2的尺寸
#proposals=utils.denorm_boxes(rpn["proposals"][0], image.shape[:2])#(1000, 4)
#captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
#            for c, s in zip(roi_class_ids[roi_positive_ixs], roi_scores[roi_positive_ixs])]
#rpnbox = rpn["proposals"][0]
#
#print(proposals[roi_positive_ixs][:5])
#coord_norm = utils.norm_boxes(proposals[roi_positive_ixs],image2.shape[:2])
#window_norm = utils.norm_boxes(window, image2.shape[:2])
#print(window)
#print(window_norm)
#coorded_norm = refineboxbywindow(window_norm,rpnbox)
#bbbox = utils.denorm_boxes(coorded_norm, image.shape[:2])[roi_positive_ixs]
#print(bbbox)
#
#visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs],
#                     refined_boxes= bbbox,
#                     captions=captions, title="ROIs After Refinement",ax=ax[1])


#########################################################
#实物边框检测

#按照窗口缩放,来调整坐标
def refineboxbywindow(window,coordinates):

    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    #按照窗口缩放坐标
    refine_coordinates = np.divide(coordinates - shift, scale)
    return refine_coordinates

#模型输出的最终检测目标结果
DetectionLayer = utils.run_graph(model,[image], [
        #(1, 100, 6) 6: 4个位置1个分类1个分数
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
],BATCH_SIZE)



##获得分类的ID
det_class_ids = DetectionLayer['detections'][0, :, 4].astype(np.int32)

det_ids = np.where(det_class_ids != 0)[0]#取出前景类不等于0的索引，
det_class_ids = det_class_ids[det_ids]#预测的分类ID
#将分类ID显示出来
print("{} detections: {}".format( len(det_ids), np.array(class_name)[det_class_ids]))

roi_scores= DetectionLayer['detections'][0, :, -1]#获得分类分数
print(roi_scores)
print(roi_scores[det_ids])

boxes_norm= DetectionLayer['detections'][0, :, :4]#
window_norm = utils.norm_boxes(window, image2.shape[:2])
boxes = refineboxbywindow(window_norm,boxes_norm)#按照窗口缩放,来调整坐标

#将坐标转化为像素坐标
refined_proposals=utils.denorm_boxes(boxes[det_ids], image.shape[:2])#(1000, 4)
captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
            for c, s in zip(det_class_ids, roi_scores[det_ids])]

visualize.draw_boxes( image, boxes=refined_proposals[det_ids],
    visibilities=[2] * len(det_ids),#统一设为2，表示用实线显示 
    captions=captions, title="Detections after NMS", ax=get_ax())

print(det_ids,refined_proposals)
print(det_class_ids)
#########################################################
#语义分割
#第三部 语义部分

#模型输出的最终检测目标结果
maskLayer = utils.run_graph(model,[image], [
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),#(1, 100, 28, 28, 81)
],BATCH_SIZE)

#按照指定的类索引，取出掩码---该掩码是每个框里的相对位移[n,28,28]
det_mask_specific = np.array([maskLayer["masks"][0, i, :, :, c] 
                              for i, c in enumerate(det_class_ids)])
print(det_mask_specific.shape)

#还原成真实大小，按照图片的框的位置来还原真实坐标(n, image.h, image.h)
true_masks = np.array([utils.unmold_mask(m, refined_proposals[i], image.shape)
                      for i, m in enumerate(det_mask_specific)])

#掩码可视化
visualize.display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
visualize.display_images(true_masks[:4] * 255, cmap="Blues", interpolation="none")

#语义分割结果可视化
t = np.transpose(true_masks,(1,2,0))
visualize.display_instances(image, refined_proposals, t, det_class_ids, 
                            class_name, roi_scores[det_ids])

##########################################################
#最终结果

results = model.detect([image], verbose=1)
#
# Visualize results
r = results[0]

#print("image", image)
#print("mask", r['masks'])
print("class_ids", r['class_ids'])
print("bbox", r['rois'])
#print("class_names", class_name)
print("scores", r['scores'])


visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_name, r['scores'])





