# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K #载入keras的后端实现
from collections import OrderedDict
import skimage.color
import skimage.io
import skimage.transform
mask_rcnn_model = __import__("8-24  mask_rcnn_model")
model =mask_rcnn_model

#Image mean (RGB)
MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
def mold_image(images):#将图片均值化
    return images.astype(np.float32) - MEAN_PIXEL

def unmold_image(normalized_images ):#将均值化的图片还原
    return (normalized_images + MEAN_PIXEL).astype(np.uint8)


#改变图片形状 ，mode 为square表明填充为正方形，大小为 max_dim
def resize_image(image, min_dim=None, max_dim=None, min_scale=None, 
                 mode="square"):#mode为pad64：支持被64整除，为crop：按照min_dim变形

    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop   

#定义函数将图片信息组合起来。
def compose_image_meta(image_id, original_image_shape, #原始图片尺寸
                       image_shape,      #image_shape转化后图片尺寸
                       window,           #转化后的图片，除去补0后剩下的坐标
                       scale, active_class_ids):
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def log(text, array=None):#输出numpy类型的对象信息
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)
 
#定义函数，运行子图
def run_graph(MaskRCNNobj, images, outputs,BATCH_SIZE, image_metas=None):
        
        model = MaskRCNNobj.keras_model#取得模型
        
        outputs = OrderedDict(outputs)#检查参数
        for o in outputs.values():
            assert o is not None

        # 通过tf.Keras的function来运行图中的一部分
        inputs = model.inputs
#        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
#            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        if image_metas is None:#将图片缩放，归一，默认返回值是window补0后的真实坐标
            molded_images, image_metas, _ = MaskRCNNobj.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        
        
        #根据图片形状获得锚点信息
        anchors = MaskRCNNobj.get_anchors(image_shape)#根据图片大小获得锚点
        
        
        #一张图片的锚点变成，batch张图片。复制batch份
        anchors = np.broadcast_to(anchors, (BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        #运行
#        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
#            model_in.append(0.)
        outputs_np = kf(model_in)

        #将结果打包成字典
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        
        for k, v in outputs_np.items():#输出结果
            log(k, v)
        return outputs_np    
############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    shape  骨干网输出的特征 256 128 64 32 16 feature_stride尺度BACKBONE_STRIDES[4, 8, 16, 32, 64]  相乘=1024
    按照BACKBONE_STRIDES个像素为单位，在图片上划分网格。得到的网格按照anchor_stride进行计算是否需要算做锚点。
    anchor_stride=1表明都要被用作计算锚点，2表明隔一个取一个网格用于计算锚点。
    每个网格第一个像素为中心点。
    边长由scales按照ratios种比例计算得到。每个中心点配上每种边长，组成一个锚点。
    """

    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()#复制了ratios个 scales--【32，32，32】
    ratios = ratios.flatten()#因为scales只有1个元素。所以不变
    
    #在以边长为scales下，将比例开方。再计算边长，另边框相对不规则一些
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    #计算像素点为单位的网格位移
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)#得到了x位移和y的位移
    #x 【【0，4，8】          y 【【0，0，0】
    #    【0，4，8】             【4，4，4】
    #    【0，4，8】】            【8，8，8】】
    
    #以每个网格第一点当作中心点，按照3种边长，为锚点大小
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    #w 【【0.5，1，2】          x 【【0，0，0】
    #    【0.5，1，2】             【4，4，4】
    #    【0.5，1，2】             【8，8，8】
    #w2  【0.5，1，2】          2  【0，0，0】
    #    【0.5，1，2】             【4，4，4】
    #    【0.5，1，2】】            【8，8，8】】

    box_centers = np.stack(#Reshape并合并中心点坐标(y, x)
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    #Reshape并合并边长(h, w)
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    #将中心点,边长转化为两个点的坐标。 (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    print(boxes[0])#因为中心点从0开始。第一个锚点的x1 y1为负数
    return boxes

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    anchors = []
    for i in range(len(scales)):#遍历不同的尺度。生成锚点
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0) #[anchor_count, (y1, x1, y2, x2)]
    
# ## Batch Slicing
# 支持批次》1的输入。为了兼容某些网络层仅有支持批次为1的输入。
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    print("batch_size",batch_size)
    for i in range(batch_size):
        print(inputs,i)
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result    




def html_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    def save_table(table):
        """Display values in a table format.
        table: an iterable of rows, and each row is an iterable of values.
        """
        html = ""
        for row in table:
            row_html = ""
            for col in row:
                row_html += "<td>{:40}</td>".format(str(col))
            html += "<tr>" + row_html + "</tr>"
        html = "<table>" + html + "</table>"

        with open('a.html','w+') as f:    	#以二进制的方式打开一个文件
            f.write(html)   	#以文本的方式写入一个用二进制打开的文件，会报错  
        print("save table ok ! in a.html")
    
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    save_table(table)


def norm_boxes(boxes, shape):
    """将像素坐标转化为标注化坐标.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = skimage.transform.resize(mask, (y2 - y1, x2 - x1), order=1, mode="constant")
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)





