# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
yolo_model = __import__("8-11  yolo_v3")
yolo_v3 = yolo_model.yolo_v3

size = 416
input_img ='timg.jpg'
output_img = 'out.jpg'
class_names = 'coco.names'
weights_file = 'yolov3.weights'
conf_threshold = 0.5 #置信度阈值
iou_threshold = 0.4  #重叠区域阈值

#定义函数：将中心点、高、宽坐标 转化为[x0, y0, x1, y1]坐标形式
def detections_boxes(detections):
    center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)
    return detections

#定义函数计算两个框的内部重叠情况（IOU）box1，box2为左上、右下的坐标[x0, y0, x1, x2]
def _iou(box1, box2):

    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    #分母加个1e-05，避免除数为 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou






#使用NMS方法，对结果去重
def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):

    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        print("shape1",shape)
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs[0]]
        print("shape2",image_pred.shape)
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result




#加载权重
def load_weights(var_list, weights_file):

    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)#跳过前5个int32
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        #找到卷积项
        if 'Conv' in var1.name.split('/')[-2]:
            # 找到BN参数项
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # 加载批量归一化参数
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                i += 4#已经加载了4个变量，指针移动4
            elif 'Conv' in var2.name.split('/')[-2]:
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                i += 1#移动指针

            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            #加载权重
            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops

#将级别结果显示在图片上
def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)
            print('{} {:.2f}%'.format(cls_names[cls], score * 100),box[:2])

def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


#加载数据集标签名称
def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def main(argv=None):
    tf.reset_default_graph()
    img = Image.open(input_img)
    img_resized = img.resize(size=(size, size))

    classes = load_coco_names(class_names)

    #定义输入占位符
    inputs = tf.placeholder(tf.float32, [None, size, size, 3])

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes), data_format='NHWC')#定义网络结构
        #加载权重
        load_ops = load_weights(tf.global_variables(scope='detector'), weights_file)

    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)

        detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
    #对10647个预测框进行去重
    filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=conf_threshold,
                                         iou_threshold=iou_threshold)

    draw_boxes(filtered_boxes, img, classes, (size, size))

    img.save(output_img)
    img.show()


if __name__ == '__main__':
    main(_)
