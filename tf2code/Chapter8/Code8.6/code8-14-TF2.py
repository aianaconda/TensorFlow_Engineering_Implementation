# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import numpy as np
from random import shuffle
annotation = __import__("code8-13-TF2")
parse_annotation = annotation.parse_annotation
ImgAugment= annotation.ImgAugment

box = __import__("code8-15-TF2")
find_match_box = box.find_match_box

DOWNSAMPLE_RATIO = 32

class BatchGenerator(object):
    def __init__(self, ann_fnames, img_dir,labels,
                 batch_size, anchors,   net_size=416,
                 jitter=True, shuffle=True):
        self.ann_fnames = ann_fnames
        self.img_dir = img_dir
        self.lable_names = labels
        self._net_size = net_size
        self.jitter = jitter
        self.anchors = create_anchor_boxes(anchors)#按照anchors的尺寸，生成坐标从00开始的框
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps_per_epoch = int(len(ann_fnames) / batch_size)
        self._epoch = 0
        self._end_epoch = False
        self._index = 0

    def next_batch(self):
        xs,ys_1,ys_2,ys_3 = [],[],[],[]
        for _ in range(self.batch_size):
            x, y1, y2, y3 = self._get()
            xs.append(x)
            ys_1.append(y1)
            ys_2.append(y2)
            ys_3.append(y3)
        if self._end_epoch == True:
            if self.shuffle:
                shuffle(self.ann_fnames)
            self._end_epoch = False
            self._epoch += 1
        return np.array(xs).astype(np.float32), np.array(ys_1).astype(np.float32), np.array(ys_2).astype(np.float32), np.array(ys_3).astype(np.float32)

    def _get(self):
        net_size = self._net_size
        #解析标注文件
        fname, boxes, coded_labels = parse_annotation(self.ann_fnames[self._index], self.img_dir, self.lable_names)
        #读取图片，并按照设置修改图片尺寸
        img_augmenter = ImgAugment(net_size, net_size, self.jitter)
        img, boxes_ = img_augmenter.imread(fname, boxes)

        #生成3种尺度的格子
        list_ys = _create_empty_xy(net_size, len(self.lable_names))
        for original_box, label in zip(boxes_, coded_labels):
            #在anchors中，找到与其面积区域最匹配的候选框max_anchor，对应的尺度索引，该尺度下的第几个锚点
            max_anchor, scale_index, box_index = _find_match_anchor(original_box, self.anchors)
            #计算在对应尺度上的中心点坐标和对应候选框的长宽缩放比例
            _coded_box = _encode_box(list_ys[scale_index], original_box, max_anchor, net_size, net_size)
            _assign_box(list_ys[scale_index], box_index, _coded_box, label)

        self._index += 1
        if self._index == len(self.ann_fnames):
            self._index = 0
            self._end_epoch = True
        
        return img/255., list_ys[2], list_ys[1], list_ys[0]

#初始化标签
def _create_empty_xy(net_size, n_classes, n_boxes=3):
    #获得最小矩阵格子
    base_grid_h, base_grid_w = net_size//DOWNSAMPLE_RATIO, net_size//DOWNSAMPLE_RATIO
    #初始化三种不同尺度的矩阵。用于存放标签
    ys_1 = np.zeros((1*base_grid_h,  1*base_grid_w, n_boxes, 4+1+n_classes)) 
    ys_2 = np.zeros((2*base_grid_h,  2*base_grid_w, n_boxes, 4+1+n_classes)) 
    ys_3 = np.zeros((4*base_grid_h,  4*base_grid_w, n_boxes, 4+1+n_classes)) 
    list_ys = [ys_3, ys_2, ys_1]
    return list_ys

def _encode_box(yolo, original_box, anchor_box, net_w, net_h):
    x1, y1, x2, y2 = original_box
    _, _, anchor_w, anchor_h = anchor_box
    #取出格子在高和宽方向上的个数
    grid_h, grid_w = yolo.shape[:2]
    
    #根据原始图片到当前矩阵的缩放比例，计算当前矩阵中，物体的中心点坐标
    center_x = .5*(x1 + x2)
    center_x = center_x / float(net_w) * grid_w 
    center_y = .5*(y1 + y2)
    center_y = center_y / float(net_h) * grid_h
    
    #计算物体相对于候选框的尺寸缩放值
    w = np.log(max((x2 - x1), 1) / float(anchor_w)) # t_w
    h = np.log(max((y2 - y1), 1) / float(anchor_h)) # t_h
    box = [center_x, center_y, w, h]#将中心点和缩放值打包返回
    return box

#找到与物体尺寸最接近的候选框
def _find_match_anchor(box, anchor_boxes):
    x1, y1, x2, y2 = box
    shifted_box = np.array([0, 0, x2-x1, y2-y1])
    max_index = find_match_box(shifted_box, anchor_boxes)
    max_anchor = anchor_boxes[max_index]
    scale_index = max_index // 3
    box_index = max_index%3
    return max_anchor, scale_index, box_index

#将具体的值放到标签矩阵里。作为真正的标签
def _assign_box(yolo, box_index, box, label):
    center_x, center_y, _, _ = box
    #向下取整，得到的就是格子的索引
    grid_x = int(np.floor(center_x))
    grid_y = int(np.floor(center_y))
    #填入所计算的数值，作为标签
    yolo[grid_y, grid_x, box_index]      = 0.
    yolo[grid_y, grid_x, box_index, 0:4] = box
    yolo[grid_y, grid_x, box_index, 4  ] = 1.
    yolo[grid_y, grid_x, box_index, 5+label] = 1.

def create_anchor_boxes(anchors):#将候选框变为box
    boxes = []
    n_boxes = int(len(anchors)/2)
    for i in range(n_boxes):
        boxes.append(np.array([0, 0, anchors[2*i], anchors[2*i+1]]))
    return np.array(boxes)




