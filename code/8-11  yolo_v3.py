# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

#定义darknet块：一个短链接加一个同尺度卷积再加一个下采样卷积
def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#正常卷积
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#正常卷积

    inputs = inputs + shortcut
    return inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    assert strides>1

    inputs = _fixed_padding(inputs, kernel_size)#外围填充0，好支持valid卷积
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding= 'VALID')

    return inputs

#对指定输入填充0
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    #inputs 【b,h,w,c】  pad  b,c不变。h和w上下左右，填充0.kernel = 3 ，则上下左右各加一趟0
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs

#定义Darknet-53 模型.返回3个不同尺度的特征
def darknet53(inputs):
    inputs = slim.conv2d(inputs, 32, 3, stride=1, padding='SAME')#正常卷积
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)#需要填充,并使用了'VALID' (-1, 208, 208, 64)
    
    inputs = _darknet53_block(inputs, 32)#darknet块
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)
    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)
    route_1 = inputs  #特征1 (-1, 52, 52, 128)

    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)
    for i in range(8):
        inputs = _darknet53_block(inputs, 256)
    route_2 = inputs#特征2  (-1, 26, 26, 256)

    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)
    for i in range(4):
        inputs = _darknet53_block(inputs, 512)#特征3 (-1, 13, 13, 512)

    return route_1, route_2, inputs#在原有的darknet53，还会跟一个全局池化。这里没有使用。所以其实是只有52层




_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

#定义候选框，来自coco数据集
_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]

#yolo检测块
def _yolo_block(inputs, filters):
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#正常卷积
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#正常卷积
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#正常卷积
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#正常卷积 
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#正常卷积
    route = inputs
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#正常卷积 
    return route, inputs

#检测层
def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    print(inputs.get_shape())
    num_anchors = len(anchors)#候选框个数
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                              activation_fn=None, biases_initializer=tf.zeros_initializer())

    shape = predictions.get_shape().as_list()
    print("shape",shape)#三个尺度的形状分别为：[1, 13, 13, 3*(5+c)]、[1, 26, 26, 3*(5+c)]、[1, 52, 52, 3*(5+c)]
    grid_size = shape[1:3]#去 NHWC中的HW
    dim = grid_size[0] * grid_size[1]#每个格子所包含的像素
    bbox_attrs = 5 + num_classes

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])#把h和w展开成dim

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])#缩放参数 32（416/13）

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]#将候选框的尺寸同比例缩小

    #将包含边框的单元属性拆分
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)#定义网格索引0,1,2...n
    grid_y = tf.range(grid_size[1], dtype=tf.float32)#定义网格索引0,1,2,...m
    a, b = tf.meshgrid(grid_x, grid_y)#生成网格矩阵 a0，a1.。。an（共M行）  ， b0，b0，。。。b0（共n个），第二行为b1

    x_offset = tf.reshape(a, (-1, 1))#展开 一共dim个
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)#连接----[dim,2]
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])#按候选框的个数复制xy（【1，n】代表第0维一次，第1维n次）

    box_centers = box_centers + x_y_offset#box_centers为0-1，x_y为具体网格的索引，相加后，就是真实位置(0.1+4=4.1，第4个网格里0.1的偏移)
    box_centers = box_centers * stride#真实尺寸像素点

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors#计算边长：hw
    box_sizes = box_sizes * stride#真实边长

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)#将转化后的结果合起来
    print(predictions.get_shape())#三个尺度的形状分别为：[1, 507（13*13*3）, 5+c]、[1, 2028, 5+c]、[1, 8112, 5+c]
    return predictions#返回预测值

#定义上采样函数
def _upsample(inputs, out_shape):
    #由于上采样的填充方式不同，tf.image.resize_bilinear会对结果影响很大
    inputs = tf.image.resize_nearest_neighbor(inputs, (out_shape[1], out_shape[2]))
    inputs = tf.identity(inputs, name='upsampled')
    return inputs


#定义yolo函数
def yolo_v3(inputs, num_classes, is_training=False, data_format='NHWC', reuse=False):

    assert data_format=='NHWC'
    
    img_size = inputs.get_shape().as_list()[1:3]#获得输入图片大小

    inputs = inputs / 255    #归一化

    #定义批量归一化参数
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  
    }

    #定义yolo网络.
    with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=data_format, reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('darknet-53'):
                route_1, route_2, inputs = darknet53(inputs)

            with tf.variable_scope('yolo-v3'):
                route, inputs = _yolo_block(inputs, 512)#(-1, 13, 13, 1024)
                #使用候选框参数来辅助识别
                detect_1 = _detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')

                
                inputs = slim.conv2d(route, 256, 1, stride=1, padding='SAME')#正常卷积 
                upsample_size = route_2.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)
                inputs = tf.concat([inputs, route_2], axis=3)

                route, inputs = _yolo_block(inputs, 256)#(-1, 26, 26, 512)
                detect_2 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')

                inputs = slim.conv2d(route, 128, 1, stride=1, padding='SAME')#正常卷积
                upsample_size = route_1.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)
                inputs = tf.concat([inputs, route_1], axis=3)

                _, inputs = _yolo_block(inputs, 128)#(-1, 52, 52, 256)

                detect_3 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')

                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')
                return detections#返回了3个尺度。每个尺度里又包含3个结果(-1, 10647（ 507 +2028 + 8112）, 5+c)




'''--------Test the scale--------'''
if __name__ == "__main__":
    tf.reset_default_graph()
    import cv2
    data = cv2.imread(  'timg.jpg' )
    data = cv2.cvtColor( data, cv2.COLOR_BGR2RGB )
    data = cv2.resize( data, ( 416, 416 ) )

    data = tf.cast( tf.expand_dims( tf.constant( data ), 0 ), tf.float32 )

    detections = yolo_v3( data,3,data_format='NHWC' )

    with tf.Session() as sess:

        sess.run( tf.global_variables_initializer() )

        print( sess.run( detections ).shape )