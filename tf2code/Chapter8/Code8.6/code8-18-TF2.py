# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import numpy as np
import cv2
box = __import__("code8-15-TF2")
boxes_to_array = box.boxes_to_array
to_minmax = box.to_minmax
BoundBox = box.BoundBox
nms_boxes = box.nms_boxes
correct_yolo_boxes = box.correct_yolo_boxes
darknet53 = __import__("code8-16-TF2")
Darknet53 = darknet53.Darknet53
weights = __import__("code8-19-TF2")
WeightReader = weights.WeightReader
yolohead = __import__("code8-17-TF2")
Headnet = yolohead.Headnet

layers = tf.keras.layers

# Yolo v3
class Yolonet(tf.keras.Model):
    def __init__(self, n_classes=80):
        
        super(Yolonet, self).__init__(name='')
        
        self.body = Darknet53()
        self.head = Headnet(n_classes)

        self.num_layers = 110
        self._init_vars()

    def load_darknet_params(self, weights_file, skip_detect_layer=False):
        weight_reader = WeightReader(weights_file)
        weight_reader.load_weights(self, skip_detect_layer)
    
    def predict(self, input_array):
        f5, f4, f3 = self.call(tf.constant(input_array.astype(np.float32)))
        return f5.numpy(), f4.numpy(), f3.numpy()

    def call(self, input_tensor, training=False):
        s3, s4, s5 = self.body(input_tensor, training)
        f5, f4, f3 = self.head(s3, s4, s5, training)
        return f5, f4, f3

    def get_variables(self, layer_idx, suffix=None):
        if suffix:
            find_name = "layer_{}/{}".format(layer_idx, suffix)
        else:
            find_name = "layer_{}/".format(layer_idx)
        variables = []
        for v in self.variables:
            if find_name in v.name:
                variables.append(v)
        return variables

    def _init_vars(self):
        sample = tf.constant(np.random.randn(1, 224, 224, 3).astype(np.float32))
        self.call(sample, training=False)
        
    def detect(self, image, anchors, net_size=416):
        image_h, image_w, _ = image.shape
        new_image = preprocess_input(image, net_size)
        # 3. predict
        yolos = self.predict(new_image)
        boxes_ = postprocess_ouput(yolos, anchors, net_size, image_h, image_w)
        
        if len(boxes_) > 0:
            boxes, probs = boxes_to_array(boxes_)
            boxes = to_minmax(boxes)
            labels = np.array([b.get_label() for b in boxes_])
        else:
            boxes, labels, probs = [], [], []
        return boxes, labels, probs

def postprocess_ouput(yolos, anchors, net_size, image_h, image_w, obj_thresh=0.5, nms_thresh=0.5):

    anchors = np.array(anchors).reshape(3, 6)
    boxes = []

    for i in range(len(yolos)):
        boxes += decode_netout(yolos[i][0], anchors[3-(i+1)], obj_thresh, net_size)

    correct_yolo_boxes(boxes, image_h, image_w)

    nms_boxes(boxes, nms_thresh)
    return boxes


def decode_netout(netout, anchors, obj_thresh, net_size, nb_box=3):

    n_rows, n_cols = netout.shape[:2]
    netout = netout.reshape((n_rows, n_cols, nb_box, -1))

    boxes = []
    for row in range(n_rows):
        for col in range(n_cols):
            for b in range(nb_box):

                x, y, w, h = _decode_coords(netout, row, col, b, anchors)
                objectness, classes = _activate_probs(netout[row, col, b, 4],#分值
                                                      netout[row, col, b, 5:],#分类
                                                      obj_thresh)

                #scale normalize                
                x /= n_cols
                y /= n_rows
                w /= net_size
                h /= net_size
                
                if objectness > obj_thresh:
                    box = BoundBox(x, y, w, h, objectness, classes)
                    boxes.append(box)

    return boxes


def _decode_coords(netout, row, col, b, anchors):
    x, y, w, h = netout[row, col, b, :4] #取出前4个坐标

    x = col + _sigmoid(x)
    y = row + _sigmoid(y)
    w = anchors[2 * b + 0] * np.exp(w)
    h = anchors[2 * b + 1] * np.exp(h)

    return x, y, w, h


def _activate_probs(objectness, classes, obj_thresh=0.3):

    # 1. sigmoid activation
    objectness_prob = _sigmoid(objectness)
    classes_probs = _sigmoid(classes)
    # 2. conditional probability
    classes_conditional_probs = classes_probs * objectness_prob
    # 3. thresholding
    classes_conditional_probs *= objectness_prob > obj_thresh
    return objectness_prob, classes_conditional_probs
    
    
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def preprocess_input(image, net_size):

    # resize the image to the new size
    preprocess_img = cv2.resize(image/255., (net_size, net_size))
    return np.expand_dims(preprocess_img, axis=0)

