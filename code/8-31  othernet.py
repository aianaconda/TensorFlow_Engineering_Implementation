# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import backend as K #载入keras的后端实现
import numpy as np
utils = __import__("8-30  mask_rcnn_utils")
mask_rcnn_model = __import__("8-29  mask_rcnn_model")

############################################################
#  Region Proposal Network (RPN)
############################################################
#构建RPN网络图结构，一共分为两部分：1计算分数，2计算边框
def rpn_graph(feature_map,#输入的特征，其w与h所围成面积的个数当作锚点的个数。
              anchors_per_location, #每个待计算锚点的网格，需要划分几种形状的矩形
              anchor_stride):#扫描网格的步长
    
    #通过一个卷积得到共享特征
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,name='rpn_conv_shared')(feature_map)

    #第一部分计算锚点的分数（前景和背景） [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    #将feature_map展开，得到[batch, anchors, 2]。anchors=feature_map的h*w*anchors_per_location 
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    #用Softmax来分类前景和背景BG/FG.结果当作分数
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    #第二部分计算锚点的边框，每个网格划分anchors_per_location种矩形框，每种4个坐标
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    #将feature_map展开，得到[batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, #扫描网格的步长
                    anchors_per_location, #每个待计算锚点的网格，需要划分几种形状的矩形
                    depth):               #输入的特征有多少个

    input_feature_map = KL.Input(shape=[None, None, depth],name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Proposal Layer
############################################################
#按照给定的框与偏移量，计算最终的框
def apply_box_deltas_graph(boxes, #[N, (y1, x1, y2, x2)]
                           deltas):#[N, (dy, dx, log(dh), log(dw))] 
    
    #转换成中心点和h，w格式
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    #计算偏移
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    #转成左上，右下两个点 y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

#将框坐标限制在0到1之间
def clip_boxes_graph(boxes, #计算完的box[N, (y1, x1, y2, x2)]
                     window):#y1, x1, y2, x2[0, 0, 1, 1]

    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(tf.keras.layers.Layer):#RPN最终处理层

    def __init__(self, proposal_count, nms_threshold,batch_size, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size


    def call(self, inputs):
        '''
        输入字段input描述
        rpn_probs: [batch, num_anchors, 2] #(bg概率, fg概率)
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, (y1, x1, y2, x2)] 
        '''
        #将前景概率值取出[Batch, num_anchors, 1]
        scores = inputs[0][:, :, 1]
        #取出位置偏移量[batch, num_anchors, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(mask_rcnn_model.RPN_BBOX_STD_DEV, [1, 1, 4])
        #取出锚点 Anchors
        anchors = inputs[2]

        #获得前6000个分值最大的数据
        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        #获取scores中索引为ix的值
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),self.batch_size)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),self.batch_size)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.batch_size, names=["pre_nms_anchors"])

        #得出最终的框坐标。[batch, N,4] (y1, x1, y2, x2),将框按照偏移缩放的数据进行计算，
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),self.batch_size,
                                  names=["refined_anchors"])


        #对出界的box进行剪辑，范围控制在 0.到1 [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,lambda x: clip_boxes_graph(x, window), self.batch_size,
                                  names=["refined_anchors_clipped"])

        # Non-max suppression算法
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")#计算nms，并获得索引
            proposals = tf.gather(boxes, indices)#在boxes中取出indices索引所指的值
            #如果proposals的个数小于proposal_count，剩下的补0
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms,self.batch_size)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)



############################################################
#  ROIAlign Layer
############################################################


#PyramidROIAlign处理
class PyramidROIAlign(tf.keras.layers.Layer):

    def __init__(self,batch_size, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.batch_size = batch_size
        
    def log2_graph(self, x):#计算log2
        return tf.log(x) / tf.log(2.0)
    
    def call(self, inputs):
        '''
        输入参数 Inputs:
        -ROIboxes(RPN结果): [batch, num_boxes, 4]，4：(y1, x1, y2, x2)。nms后得锚点坐标.num_boxes=1000
        - image_meta: [batch, (meta data)] 图片的附加信息 93
        - Feature maps: [P2, P3, P4, P5]骨干网经过fpn后的特征.每个[batch, height, width, channels]
        [(1, 256, 256, 256),(1, 128, 128, 256),(1, 64, 64, 256),(1, 32, 32, 256)]
        '''
        #获取输入参数
        ROIboxes = inputs[0]#(1, 1000, 4) 
        image_meta = inputs[1]#(1, 93)
        feature_maps = inputs[2:]

        #将锚点坐标提出来
        y1, x1, y2, x2 = tf.split(ROIboxes, 4, axis=2)#[batch, num_boxes, 4]
        h = y2 - y1
        w = x2 - x1

        
        ###############################在这1000个ROI里，按固定算法匹配到不同level的特征。
        #获得图片形状
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        #因为h与w是标准化坐标。其分母已经被除了tf.sqrt(image_area)。
        #这里再除以tf.sqrt(image_area)分之1，是为了变为像素坐标
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum( 2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # 每个roi按照自己的区域去对应的特征里截取内容，并resize成指定的7*7大小. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            #equal会返回一个true false的（1，1000），where返回其中为true的索引[[0,1],[0,4],,,[0,200]]
            ix = tf.where(tf.equal(roi_level, level),name="ix")#(828, 2)

            
            #在多维上建立索引取值[?,4](828, 4)
            level_boxes = tf.gather_nd(ROIboxes, ix,name="level_boxes")#在(1, 1000, 4)上按照[[0,1],[0,4],,,[0,200]]取值 

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)#(828, )，【0，0，0，0，0，】如果批次为2，就是[000...111]

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            #下面两个值，是ROIboxes中按照不同尺度划分好的索引，对于该尺度特征中的批次索引，不希望有变化。所以停止梯度
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)


            #结果: [batch * num_boxes, pool_height, pool_width, channels]
            #feature_maps [(1, 256, 256, 256),(1, 128, 128, 256),(1, 64, 64, 256),(1, 32, 32, 256)]
            #box_indices一共level_boxes个。指定level_boxes中的第几个框，作用于feature_maps中的第几个图片
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"))

        #1000个roi都取到了对应的内容，将它们组合起来。( 1000, 7, 7, 256) 
        pooled = tf.concat(pooled, axis=0)#其中的顺序是按照level来的需要重新排列成原来ROIboxes顺序

        #重新排列成原来ROIboxes顺序
        box_to_level = tf.concat(box_to_level, axis=0)#按照选取level的顺序
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],axis=1)#[1000，3] 3([xi] range)
                                 

        #取出头两个批次+序号（1000个），每个值代表原始roi展开的索引了。
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1] #保证一个批次在100000以内
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(#按照索引排序，
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)#将roi中的顺序对应到pooled中的索引取出来
        pooled = tf.gather(pooled, ix)#按照索引从pooled中取出的框，就是原始顺序了。

        #加上批次维度，并返回
        pooled = tf.expand_dims(pooled, 0)#应该用reshape
        #pooled = KL.Reshape([self.batch_size,-1, self.pool_shape, self.pool_shape, mask_rcnn_model.FPN_FEATURE], name="pooled")(pooled)
        #pooled = tf.reshape(pooled, [self.batch_size,1000, self.pool_shape, self.pool_shape, mask_rcnn_model.FPN_FEATURE])
        return pooled
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )
    
############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, batch_size, train_bn=True,
                         fc_layers_size=1024):


    #ROIAlign层 Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(batch_size,[pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    #用卷积替代两个1024全连接网络
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    #1*1卷积，代替第二个全连接
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    #共享特征，用于计算分类和边框
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),name="pool_squeeze")(x)

    #（1）计算分类
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    #（2）计算边框坐标BBox（偏移和缩放量）
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    print(s, num_classes, 4)
    #mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
    mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)


    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox    
 
############################################################
#  Detection Layer
############################################################
#实物边框检测，返回最终的标准化区域坐标[batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]
class DetectionLayer(tf.keras.layers.Layer):

    def __init__(self,batch_size,  **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        
    def call(self, inputs):#输入：rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta
        #提取参数
        rois,mrcnn_class,mrcnn_bbox,image_meta = inputs

        #解析图片附加信息
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        #window为pading后，真实图片的像素坐标，将其转化为标准坐标
        window = norm_boxes_graph(m['window'], image_shape[:2])
        
        #根据分类信息，对原始roi进行再一次的过滤。得到DETECTION_MAX_INSTANCES个。不足的补0
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z),
            self.batch_size)

        #将标准化坐标及过滤后的结果 reshape后返回。
        return tf.reshape(
            detections_batch,
            [self.batch_size, mask_rcnn_model.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, mask_rcnn_model.DETECTION_MAX_INSTANCES, 6)

#将坐标按照图片大小，转化为标准化坐标
def norm_boxes_graph(boxes, #像素坐标(y1, x1, y2, x2)
                     shape):#像素边长(height, width)
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)#标准化坐标[..., (y1, x1, y2, x2)]

#分类器结果的最终处理函数，返回剪辑后的标准坐标与去重后的分类结果
def refine_detections_graph(rois, probs, deltas, window):
   
    #取出每个ROI的 Class IDs
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    
    #取出每个ROI的class 索引
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)#根据索引获得分数
    
    deltas_specific = tf.gather_nd(deltas, indices)#根据索引获得box区域坐标(待修正的偏差)

    #将偏差应用到rois框中
    refined_rois = apply_box_deltas_graph( rois, deltas_specific * mask_rcnn_model.BBOX_STD_DEV)
    #对出界的框进行剪辑
    refined_rois = clip_boxes_graph(refined_rois, window)

    #取出前景的类索引（将背景类过滤掉）
    keep = tf.where(class_ids > 0)[:, 0]
    #从前景类里，再将分数小于DETECTION_MIN_CONFIDENCE的过滤掉
    if mask_rcnn_model.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= mask_rcnn_model.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    #根据剩下的keep索引取出对应的值
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):#定义nms函数，对每个类做去重
        
        #找出类别为class_id 的索引
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        
        #对该类的roi按照阈值DETECTION_NMS_THRESHOLD进行区域去重，最多获得DETECTION_MAX_INSTANCES个结果，
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=mask_rcnn_model.DETECTION_MAX_INSTANCES,
                iou_threshold=mask_rcnn_model.DETECTION_NMS_THRESHOLD)
        #将去重后的索引转化为roi中的索引
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        #数据对齐，当去重后的个数小于DETECTION_MAX_INSTANCES时，对其补-1.
        gap = mask_rcnn_model.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        #将形状统一变为[mask_rcnn_model.DETECTION_MAX_INSTANCES]，并返回
        class_keep.set_shape([mask_rcnn_model.DETECTION_MAX_INSTANCES])
        return class_keep

    #对每个class IDs做去重操作。
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    #将list结果中的元素合并到一个数组里。并删掉-1的值
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    #计算交集。没用
#    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
#                                    tf.expand_dims(nms_keep, 0))
#    keep = tf.sparse_tensor_to_dense(keep)[0]
    keep = nms_keep#改成这样
    #nms之后，根据剩下的keep索引取出对应的值，将总是控制在DETECTION_MAX_INSTANCES之内
    roi_count = mask_rcnn_model.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)#keep个数小于DETECTION_MAX_INSTANCES


    #拼接输出结果[N, (y1, x1, y2, x2, class_id, score)]
    detections = tf.concat([ tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    #数据对齐，不足DETECTION_MAX_INSTANCES的补0，并返回。
    gap = mask_rcnn_model.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections    

#语义分割
def build_fpn_mask_graph(rois,#目标实物检测结果，标准坐标[batch, num_rois, (y1, x1, y2, x2)] 
                         feature_maps,#骨干网之后的fpn特征[P2, P3, P4, P5]
                         image_meta,
                         pool_size, num_classes,batch_size, train_bn=True):
    """
    返回: Masks [batch, roi_count, height, width, num_classes]
    """
    #ROIAlign 最终统一池化的大小为14 
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(batch_size,[pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)#(1, ?, 14, 14, 256)
    
    #使用反卷积进行上采样
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)#(1, ?, 28, 28, 256)
    #用卷积代替全连接
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x
























############################################################
#  Data Formatting
############################################################



def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros
