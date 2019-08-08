"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import os
import random
import datetime
import re
import math
import logging
import numpy as np
import skimage.transform
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K #载入keras的后端实现
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM

utils = __import__("8-25  mask_rcnn_utils")
log = utils.log
compose_image_meta = utils.compose_image_meta
othernet =  __import__("8-27  othernet")
build_rpn_model = othernet.build_rpn_model
ProposalLayer= othernet.ProposalLayer
fpn_classifier_graph = othernet.fpn_classifier_graph
DetectionLayer = othernet.DetectionLayer
build_fpn_mask_graph = othernet.build_fpn_mask_graph
parse_image_meta_graph = othernet.parse_image_meta_graph


# 由于MNS算法的版本不同，锁定在 TensorFlow 1.8+ 
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.8")

#定义全局输入图片大小（二选一）,图片会被下采样6次。必须能够被2的6次方整除
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024

IMAGE_DIM = IMAGE_MAX_DIM  #选择1024
IMAGE_RESIZE_MODE = "square"#统一成IMAGE_MAX_DIM

# 图片resize时，定义的最小的缩放范围.0代表不进行最小缩放范围限制
IMAGE_MIN_SCALE = 0

BACKBONE = "resnet101"     #主干网络使用resnet

#骨干网返回的每一层特征，对原始图片的缩小比例.代表着输出特征的5种尺度
#在计算锚点时，BACKBONE_STRIDES的每个元素，代表按照该像素值划分网格，
#骨干网的输出形状即为256 128 64 32 16,代表输出的网格个数为256 128 64 32 16
BACKBONE_STRIDES = [4, 8, 16, 32, 64]

#扫描网格的步长。按照该步长获取网格，用于计算锚点。网格中的第一个像素坐标被当作锚点的中心点
RPN_ANCHOR_STRIDE = 1

#每个锚点的边长初始值
RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

#锚点的边长比例(width/height)，将初始值和边长比例一起计算，得到锚点的真实边长。
RPN_ANCHOR_RATIOS = [0.5, 1, 2]

#训练RPN网络时选取锚点的个数
RPN_TRAIN_ANCHORS_PER_IMAGE = 256
#在训练过程中，将选取多少个ROI放到fpn层
TRAIN_ROIS_PER_IMAGE = 200
#训练过程中选取的正向ROI比例
# Percent of positive ROIs used to train classifier/mask heads
ROI_POSITIVE_RATIO = 0.33

#对应与训练或是使用时，RPN网络最终需要最大保留多少个ROI
POST_NMS_ROIS_TRAINING = 2000
POST_NMS_ROIS_INFERENCE = 1000
RPN_NMS_THRESHOLD = 0.7
FPN_FEATURE = 256 #特征金字塔层的深度
DETECTION_MAX_INSTANCES = 100#fpn最终检测的实例个数
#在制作样本的标签时，从一张图中，最多只读取100个实例
MAX_GT_INSTANCES = 100
#分类时的置信度阈值
DETECTION_MIN_CONFIDENCE = 0.7
#检测时的Non-maximum suppression阈值
DETECTION_NMS_THRESHOLD = 0.3

# Pooled ROIs
POOL_SIZE = 7#金字塔对齐池化后的ROI形状
MASK_POOL_SIZE = 14
MASK_SHAPE = [28, 28]


#RPN和最终检测的边界框细化标准偏差 Bounding box refinement standard deviation for RPN and final detections.
RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

#是否对掩码进行压缩
USE_MINI_MASK = True
MINI_MASK_SHAPE = (56, 56)  # 压缩后的掩码大小(height, width)



############################################################
#  Resnet Graph
############################################################

#计算resnet返回的形状
def compute_backbone_shapes( image_shape):
    
    returnshape  = [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))] for stride in BACKBONE_STRIDES]
    
    return np.array( returnshape)

#resnet中的identity_block(不带卷积的短链接):kernel_size为第二层卷积核大小。filters为每层卷积核个数.stage和block用于命名
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    
    nb_filter1, nb_filter2, nb_filter3 = filters#解析出每层卷积核个数
    conv_name_base = 'res' + str(stage) + block + '_branch'#为卷积层命名
    bn_name_base = 'bn' + str(stage) + block + '_branch'#为BN层命名

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])#短链接
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

#resnet中的conv_block(带卷积的短链接):strides为第一层的步长 ，进行了下采样,所以带卷积的短链接时也得下采样
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #第一层1*1 卷积
    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    #第二层，按照指定卷积核卷积
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    #第三层，1*1 卷积
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    #带卷积的短链接
    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    
    return x

#组建残差网络，支持resnet50 or resnet101 两种。  stage5是否将第5特征层的结果输出
def resnet_graph(input_image, architecture, stage5=False, train_bn=True):

    assert architecture in ["resnet50", "resnet101"]
    #第1特征层
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    #第2特征层
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    #第3特征层
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    #第4特征层
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # 第5特征层
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  MaskRCNN Class
############################################################
class MaskRCNN():    #内部封装的keras_model为真正模型
    def __init__(self, mode, model_dir,num_class,batch_size):#初始化
        """
        mode: 可以是 "training" 或 "inference" 两种模式
        model_dir: 保存模型的路径
        """
        assert mode == 'inference'
        self.mode = mode
        self.num_class = num_class
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode)
        
    def mold_inputs(self, images):#输入图片预处理
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            
            #window是缩放后有效图片的坐标
            #scale是缩放比例
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=IMAGE_MIN_DIM,
                min_scale=IMAGE_MIN_SCALE,
                max_dim=IMAGE_MAX_DIM,
                mode=IMAGE_RESIZE_MODE)
            molded_image = utils.mold_image(molded_image)#均值化
            
            #把图片配套的信息也打包好
            image_meta = utils.compose_image_meta(  
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.num_class], dtype=np.int32))
            #将信息添加到列表
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        
        #转成np数组
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        
        return molded_images, image_metas, windows

    def build(self, mode):#构建Mask R-CNN架构

        #检查尺寸合法性
        h, w = IMAGE_DIM,IMAGE_DIM;
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("必须要被2的6次方整除.例如： 256, 320, 384, 448, 512, ... etc. ")
         
        input_image = KL.Input( shape=[None, None, 3], name="input_image")#定义输入节点
        
        if mode == "inference":
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")#将全局的锚点框输入

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        #构建骨干网络。返回最后5层的特征（5种尺度）
        #不使用BN，因为批次=2，非常小
        _, C2, C3, C4, C5 = resnet_graph(input_image, BACKBONE,stage5=True, train_bn=False)
        
        # Top-down Layers
        #特征金字塔层fpn。  最大的c1不要了。每层256个
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                        KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)]  )
        P3 = KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                        KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)]  )
        P2 = KL.Add(name="fpn_p2add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                        KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)]  )
        
    # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(FPN_FEATURE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(FPN_FEATURE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(FPN_FEATURE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(FPN_FEATURE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        
        rpn_feature_maps = [P2, P3, P4, P5, P6]#用于rpn使用,P5是最全的特征，对p5进行下采样生成p6
        mrcnn_feature_maps = [P2, P3, P4, P5]#用于classifier heads 使用
        
        ###########################################
        #准备RPN网络
        

        # RPN Model 该模型会生成前后景 前后景概率 box  output_names
        rpn = build_rpn_model(RPN_ANCHOR_STRIDE, len(RPN_ANCHOR_RATIOS), FPN_FEATURE)#每个尺度的特征都是256
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))#将resnet特征分别放到rpn网络中，输出放到layer_outputs里面
        
        #将结果 Concatenate 到一起
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        #需要保留ROI的个数
        proposal_count = POST_NMS_ROIS_TRAINING if mode == "training" else POST_NMS_ROIS_INFERENCE
         
        # Anchors
        if mode == "inference":
            anchors = input_anchors
        ##返回nms去重后，前景分数最大的n个ROI
        rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=RPN_NMS_THRESHOLD,batch_size=self.batch_size,
                                 name="ROI")([rpn_class, rpn_bbox, anchors])
###################################################
        #下面数字意义：image_id=1 original_image_shape=3 image_shape=3 坐标=4 缩放=1
        img_meta_size = 1 + 3 + 3 + 4 + 1 + self.num_class 		#定义图片附加信息
 
        input_image_meta = KL.Input(shape=[img_meta_size], name="input_image_meta")#定义图片附加信息
        
        #fpn网络对rpn_rois区域与特征数据 mrcnn_feature_maps进行计算。识别出分类、边框和掩码
        if mode == "inference":
            # Network Heads
            # 对rpn_rois区域内的mrcnn_feature_maps做分类，并微调box  Proposal classifier and BBox regressor heads
            #mrcnn_class分类  mrcnn_bbox中心点长宽变化量
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     POOL_SIZE, self.num_class ,self.batch_size,
                                     train_bn=False,#不用bn
                                     fc_layers_size=1024)#全连接层1024个节点

            #输出标准化坐标 [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
            #将rpn_rois与mrcnn_class, mrcnn_bbox组合起来，算出真实的box
            detections = DetectionLayer( batch_size= self.batch_size, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            #像素分割
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)#取出box坐标
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,MASK_POOL_SIZE,#14
                                              self.num_class,self.batch_size,train_bn=False)#不用bn

            model = KM.Model([input_image, input_image_meta, input_anchors],#输入
        [detections, mrcnn_class, mrcnn_bbox,mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],#输出
                             name='mask_rcnn')

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = "maskrcnn"
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
#        print("load model",filepath)
#         #self.keras_model=tf.keras.models.load_model(filepath)
#        self.keras_model.load_weights(filepath)
#        
#        return
        
        
        import h5py
        from tensorflow.python.keras.engine import saving
#        # Conditional import to support versions of Keras before 2.2
#        # TODO: remove in about 6 months (end of 2018)
#        try:
#            from keras.engine import saving
#        except ImportError:
#            # Keras before 2.2 used the 'topology' namespace.
#            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            #tf.keras.engine.saving.load_weights_from_hdf5_group_by_name(f, layers)
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            #tf.keras.engine.saving.load_weights_from_hdf5_group(f, layers)
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)


    #梯度剪辑
    GRADIENT_CLIP_NORM = 5.0
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50
    
    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            tf.keras.regularizers.l2(WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            print("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            "maskrcnn", now))

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            "maskrcnn"))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")





    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]
        print("VVVVVVVVVVVVVVVVVV",boxes[:3])
        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        print("window",window)
        print("image_shape[:2]",image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])
        print("original_image_shape[:2]",original_image_shape[:2])
        print(boxes[:3])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):#使用模型进行检测
        """使用模型进行检测
        输入： images
        输出：字典类型。包括如下内容
        rois: 检测框[N, (y1, x1, y2, x2)]
        class_ids: 类别[N]
        scores: 分数[N]
        masks: 掩码[H, W, N] 
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len( images) == self.batch_size, "len(images) must be equal to BATCH_SIZE"

        if verbose:#是否输出信息
            print("Processing {} images".format(len(images)))


        #图片预处理（统一大小，并返回图片附加信息）
        molded_images, image_metas, windows = self.mold_inputs(images)

        #验证尺寸
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        #生成锚点
        anchors = self.get_anchors(image_shape)
        #复制锚点到批次
        anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        #运行模型进行图片分析
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        #处理分析结果
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results
    
    def get_anchors(self, image_shape):
        """根据指定图片大小生成锚点."""
        backbone_shapes = compute_backbone_shapes( image_shape)
        # 缓存锚点
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            #生成锚点
            a = utils.generate_pyramid_anchors(RPN_ANCHOR_SCALES,RPN_ANCHOR_RATIOS,
                backbone_shapes,BACKBONE_STRIDES,RPN_ANCHOR_STRIDE)
            self.anchors = a
            #设为标准坐标
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.batch_size,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 5000:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

#    def run_graph(self, images, outputs, image_metas=None):
#        """Runs a sub-set of the computation graph that computes the given
#        outputs.
#
#        image_metas: If provided, the images are assumed to be already
#            molded (i.e. resized, padded, and normalized)
#
#        outputs: List of tuples (name, tensor) to compute. The tensors are
#            symbolic TensorFlow tensors and the names are for easy tracking.
#
#        Returns an ordered dict of results. Keys are the names received in the
#        input and values are Numpy arrays.
#        """
#        model = self.keras_model
#
#        # Organize desired outputs into an ordered dict
#        outputs = OrderedDict(outputs)
#        for o in outputs.values():
#            assert o is not None
#
#        # Build a Keras function to run parts of the computation graph
#        inputs = model.inputs
#        #print("inputs______________",inputs)
#        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
#            inputs += [K.learning_phase()]
#        kf = K.function(model.inputs, list(outputs.values()))
#
#        # Prepare inputs
#        if image_metas is None:
#            molded_images, image_metas, _ = self.mold_inputs(images)#将图片缩放，归一，默认返回值是window补0后的真实坐标
#        else:
#            molded_images = images
#        image_shape = molded_images[0].shape
#        
#        #print("molded_images______________",molded_images)
#        
#        # Anchors
#        anchors = self.get_anchors(image_shape)#根据图片大小获得锚点
#        # Duplicate across the batch dimension because Keras requires it
#        # TODO: can this be optimized to avoid duplicating the anchors?
#        #一张图片的锚点变成，batch张图片。复制batch份
#        anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)
#        model_in = [molded_images, image_metas, anchors]
#
#        # Run inference
#        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
#            model_in.append(0.)
#        outputs_np = kf(model_in)
#
#        # Pack the generated Numpy arrays into a a dict and log the results.
#        outputs_np = OrderedDict([(k, v)
#                                  for k, v in zip(outputs.keys(), outputs_np)])
#        for k, v in outputs_np.items():
#            log(k, v)
#        return outputs_np







############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

#
#def rpn_bbox_loss_graph( target_bbox, rpn_match, rpn_bbox,batch_size):
#    """Return the RPN bounding box loss graph.
#
#   
#    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
#        Uses 0 padding to fill in unsed bbox deltas.
#    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
#               -1=negative, 0=neutral anchor.
#    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
#    """
#    # Positive anchors contribute to the loss, but negative and
#    # neutral anchors (match value of 0 or -1) don't.
#    rpn_match = K.squeeze(rpn_match, -1)
#    indices = tf.where(K.equal(rpn_match, 1))
#
#    # Pick bbox deltas that contribute to the loss
#    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
#
#    # Trim target bounding box deltas to the same length as rpn_bbox.
#    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
#    target_bbox = batch_pack_graph(target_bbox, batch_counts,  batch_size)
#
#    # TODO: use smooth_l1_loss() rather than reimplementing here
#    #       to reduce code duplication
#    diff = K.abs(target_bbox - rpn_bbox)
#    less_than_one = K.cast(K.less(diff, 1.0), "float32")
#    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
#
#    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
#    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=IMAGE_MIN_DIM,
        min_scale=IMAGE_MIN_SCALE,
        max_dim=IMAGE_MAX_DIM,
        mode=IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks,num_class):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(TRAIN_ROIS_PER_IMAGE * ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((TRAIN_ROIS_PER_IMAGE,
                       num_class, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((TRAIN_ROIS_PER_IMAGE,MASK_SHAPE[0],MASK_SHAPE[1], num_class),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros([IMAGE_DIM,IMAGE_DIM], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(skimage.transform.resize(
                    class_mask, (gt_h, gt_w), order=1, mode="constant")).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = skimage.transform.resize(m, MASK_SHAPE, order=1, mode="constant")
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks

#
#def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes):
#    """Given the anchors and GT boxes, compute overlaps and identify positive
#    anchors and deltas to refine them to match their corresponding GT boxes.
#
#    anchors: [num_anchors, (y1, x1, y2, x2)]
#    gt_class_ids: [num_gt_boxes] Integer class IDs.
#    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
#
#    Returns:
#    rpn_match: [N] (int32) matches between anchors and GT boxes.
#               1 = positive anchor, -1 = negative anchor, 0 = neutral（iou在0.3和0.7之间）
#    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
#    """
#    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
#    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
#    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
#    #其实rpn_bbox可以砍掉一般。因为只有放了一半的正锚点
#    rpn_bbox = np.zeros((RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
#
#    # Handle COCO crowds
#    # A crowd box in COCO is a bounding box around several instances. Exclude
#    # them from training. A crowd box is given a negative class ID.
#    crowd_ix = np.where(gt_class_ids < 0)[0]
#    if crowd_ix.shape[0] > 0:
#        # Filter out crowds from ground truth class IDs and boxes
#        non_crowd_ix = np.where(gt_class_ids > 0)[0]
#        crowd_boxes = gt_boxes[crowd_ix]
#        gt_class_ids = gt_class_ids[non_crowd_ix]
#        gt_boxes = gt_boxes[non_crowd_ix]
#        # Compute overlaps with crowd boxes [anchors, crowds]
#        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
#        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
#        no_crowd_bool = (crowd_iou_max < 0.001)
#    else:
#        # All anchors don't intersect a crowd
#        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)
#
#    # Compute overlaps [num_anchors, num_gt_boxes]  每个值都是面积重叠的比例
#    overlaps = utils.compute_overlaps(anchors, gt_boxes)
#
#    # Match anchors to GT Boxes
#    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
#    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
#    # Neutral anchors are those that don't match the conditions above,
#    # and they don't influence the loss function.
#    # However, don't keep any GT box unmatched (rare, but happens). Instead,
#    # match it to the closest anchor (even if its max IoU is < 0.3).
#    #
#    # 1. Set negative anchors first. They get overwritten below if a GT box is
#    # matched to them. Skip boxes in crowd areas.
#    anchor_iou_argmax = np.argmax(overlaps, axis=1)
#    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
#    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1#将Iou《0.3的设为-1
#    # 2. Set an anchor for each GT box (regardless of IoU value).
#    # TODO: If multiple anchors have the same IoU match all of them
#    gt_iou_argmax = np.argmax(overlaps, axis=0)
#    rpn_match[gt_iou_argmax] = 1#将锚点中，对应与任何一个bbox的Iou最大的值都设为1
#    # 3. Set anchors with high overlap as positive.
#    rpn_match[anchor_iou_max >= 0.7] = 1#将将Iou>= 0.7的设为
#    #要么充分重叠，要么不充分重叠。  介于二者之前的都是0
#
#    # Subsample to balance positive and negative anchors
#    # Don't let positives be more than half the anchors
#    #总共256个正负锚点框。正负锚点有超过半数的，通过随机值将其去掉。
#    ids = np.where(rpn_match == 1)[0]
#    extra = len(ids) - (RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
#    if extra > 0:
#        # Reset the extra ones to neutral
#        ids = np.random.choice(ids, extra, replace=False)
#        rpn_match[ids] = 0
#    # Same for negative proposals
#    ids = np.where(rpn_match == -1)[0]
#    extra = len(ids) - (RPN_TRAIN_ANCHORS_PER_IMAGE -
#                        np.sum(rpn_match == 1))
#    if extra > 0:
#        # Rest the extra ones to neutral
#        ids = np.random.choice(ids, extra, replace=False)
#        rpn_match[ids] = 0
#
#    # For positive anchors, compute shift and scale needed to transform them
#    # to match the corresponding GT boxes.
#    ids = np.where(rpn_match == 1)[0]
#    ix = 0  # index into rpn_bbox
#    # TODO: use box_refinement() rather than duplicating the code here
#    for i, a in zip(ids, anchors[ids]):
#        # Closest gt box (it might have IoU < 0.7)
#        gt = gt_boxes[anchor_iou_argmax[i]]
#
#        # Convert coordinates to center plus width/height.
#        # GT Box
#        #计算bbox的高、宽、中心点
#        gt_h = gt[2] - gt[0]
#        gt_w = gt[3] - gt[1]
#        gt_center_y = gt[0] + 0.5 * gt_h
#        gt_center_x = gt[1] + 0.5 * gt_w
#        
#        #计算Anchor的高、宽、中心点
#        a_h = a[2] - a[0]
#        a_w = a[3] - a[1]
#        a_center_y = a[0] + 0.5 * a_h
#        a_center_x = a[1] + 0.5 * a_w
#
#        # Compute the bbox refinement that the RPN should predict.
#        #计算需要预测的中心点偏移比例及高、宽缩放比例
#        rpn_bbox[ix] = [
#            (gt_center_y - a_center_y) / a_h,
#            (gt_center_x - a_center_x) / a_w,
#            np.log(gt_h / a_h),
#            np.log(gt_w / a_w),
#        ]
#        # Normalize
#        rpn_bbox[ix] /= RPN_BBOX_STD_DEV
#        ix += 1
#
#    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])#计算每个bbox需要生成的随机框
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        #在gt_boxes范围内生成rois_per_box个随机框。要求边长都得大于threshold
        #如果一次每生成对。就再来一次，直到生成对为止
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            #从边长大于1的框里面取出rois_per_box个。如果不足，shape[0]将小于rois_per_box
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=threshold][:rois_per_box]
            #没生成够，就再来一次
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        
        #将该bbox生成的随机框填入rois
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    #剩余10%的随机框存放空间
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    
    #按照整个图片进行随机框生成
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois#将最后的随机填入
    return rois

#
#def data_generator(dataset,  shuffle=True, augment=False, augmentation=None,
#                   random_rois=0, batch_size=1, detection_targets=False,
#                   no_augmentation_sources=None,num_class = 0):
#    """A generator that returns images and corresponding target class ids,
#    bounding box deltas, and masks.
#
#    dataset: The Dataset object to pick data from
#    shuffle: If True, shuffles the samples before every epoch
#    augment: (deprecated. Use augmentation instead). If true, apply random
#        image augmentation. Currently, only horizontal flipping is offered.
#    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
#        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
#        right/left 50% of the time.
#    random_rois: If > 0 then generate proposals to be used to train the
#                 network classifier and mask heads. Useful if training
#                 the Mask RCNN part without the RPN.
#    batch_size: How many images to return in each call
#    detection_targets: If True, generate detection targets (class IDs, bbox
#        deltas, and masks). Typically for debugging or visualizations because
#        in trainig detection targets are generated by DetectionTargetLayer.
#    no_augmentation_sources: Optional. List of sources to exclude for
#        augmentation. A source is string that identifies a dataset and is
#        defined in the Dataset class.
#
#    Returns a Python generator. Upon calling next() on it, the
#    generator returns two lists, inputs and outputs. The contents
#    of the lists differs depending on the received arguments:
#    inputs list:
#    - images: [batch, H, W, C]
#    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
#    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
#    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
#    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
#    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
#    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
#                are those of the image unless use_mini_mask is True, in which
#                case they are defined in MINI_MASK_SHAPE.
#
#    outputs list: Usually empty in regular training. But if detection_targets
#        is True then the outputs list contains target class_ids, bbox deltas,
#        and masks.
#    """
#    b = 0  # batch item index
#    image_index = -1
#    image_ids = np.copy(dataset.image_ids)#所有图片的id数组
#    error_count = 0
#    no_augmentation_sources = no_augmentation_sources or []
#
#    # Anchors
#    # [anchor_count, (y1, x1, y2, x2)]
#    backbone_shapes = compute_backbone_shapes( [IMAGE_DIM,IMAGE_DIM])#【256 128 64 32 16】---1024/[4, 8, 16, 32, 64]
#    anchors = utils.generate_pyramid_anchors(RPN_ANCHOR_SCALES,
#                                             RPN_ANCHOR_RATIOS,
#                                             backbone_shapes,
#                                             BACKBONE_STRIDES,
#                                             RPN_ANCHOR_STRIDE)
#
#    # Keras requires a generator to run indefinitely.
#    while True:
#        try:
#            # Increment index to pick next image. Shuffle if at the start of an epoch.
#            image_index = (image_index + 1) % len(image_ids)
#            if shuffle and image_index == 0:
#                np.random.shuffle(image_ids)
#
#            # Get GT bounding boxes and masks for image.
#            image_id = image_ids[image_index]#娶一个索引ID
#
#            # If the image source is not to be augmented pass None as augmentation
#            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
#                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
#                load_image_gt(dataset, image_id, augment=augment,
#                              augmentation=None,
#                              use_mini_mask=USE_MINI_MASK)
#            else:
#                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
#                    load_image_gt(dataset,image_id, augment=augment,
#                                augmentation=augmentation,
#                                use_mini_mask=USE_MINI_MASK)
#
#            # Skip images that have no instances. This can happen in cases
#            # where we train on a subset of classes and the image doesn't
#            # have any of the classes we care about.
#            if not np.any(gt_class_ids > 0):
#                continue
#
#            # RPN Targets
#            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
#                                                    gt_class_ids, gt_boxes)
#
#            # Mask R-CNN Targets
#            if random_rois:
#                #根据bbox和image，生成了random_rois个随机框
#                rpn_rois = generate_random_rois(image.shape, random_rois, gt_class_ids, gt_boxes)
#                
#                if detection_targets:
#                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
#                        build_detection_targets(
#                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, num_class)
#
#            # Init batch arrays
#            if b == 0:
#                batch_image_meta = np.zeros( (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
#                batch_rpn_match = np.zeros(  [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
#                batch_rpn_bbox = np.zeros(  [batch_size, RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
#                batch_images = np.zeros( (batch_size,) + image.shape, dtype=np.float32)
#                batch_gt_class_ids = np.zeros( (batch_size, MAX_GT_INSTANCES), dtype=np.int32)
#                batch_gt_boxes = np.zeros( (batch_size, MAX_GT_INSTANCES, 4), dtype=np.int32)
#                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
#                     MAX_GT_INSTANCES), dtype=gt_masks.dtype)
#                if random_rois:
#                    batch_rpn_rois = np.zeros(
#                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
#                    if detection_targets:
#                        batch_rois = np.zeros(
#                            (batch_size,) + rois.shape, dtype=rois.dtype)
#                        batch_mrcnn_class_ids = np.zeros(
#                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
#                        batch_mrcnn_bbox = np.zeros(
#                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
#                        batch_mrcnn_mask = np.zeros(
#                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
#
#            # If more instances than fits in the array, sub-sample from them.
#            if gt_boxes.shape[0] > MAX_GT_INSTANCES:
#                ids = np.random.choice(
#                    np.arange(gt_boxes.shape[0]), MAX_GT_INSTANCES, replace=False)
#                gt_class_ids = gt_class_ids[ids]
#                gt_boxes = gt_boxes[ids]
#                gt_masks = gt_masks[:, :, ids]
#
#            # Add to batch
#            batch_image_meta[b] = image_meta
#            batch_rpn_match[b] = rpn_match[:, np.newaxis]
#            batch_rpn_bbox[b] = rpn_bbox
#            batch_images[b] = mold_image(image.astype(np.float32))
#            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
#            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
#            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
#            if random_rois:
#                batch_rpn_rois[b] = rpn_rois
#                if detection_targets:
#                    batch_rois[b] = rois
#                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
#                    batch_mrcnn_bbox[b] = mrcnn_bbox
#                    batch_mrcnn_mask[b] = mrcnn_mask
#            b += 1
#
#            # Batch full?
#            if b >= batch_size:
#                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
#                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
#                outputs = []
#
#                if random_rois:
#                    inputs.extend([batch_rpn_rois])
#                    if detection_targets:
#                        inputs.extend([batch_rois])
#                        # Keras requires that output and targets have the same number of dimensions
#                        batch_mrcnn_class_ids = np.expand_dims(
#                            batch_mrcnn_class_ids, -1)
#                        outputs.extend(
#                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])
#
#                yield inputs, outputs
#
#                # start a new batch
#                b = 0
#        except (GeneratorExit, KeyboardInterrupt):
#            raise
#        except:
#            # Log it and skip the image
#            logging.exception("Error processing image {}".format(
#                dataset.image_info[image_id]))
#            error_count += 1
#            if error_count > 5:
#                raise
#






############################################################
#  Miscellenous Graph Functions
############################################################



def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)





def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
