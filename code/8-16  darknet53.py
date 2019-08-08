# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf

layers = tf.keras.layers


class Darknet53(tf.keras.Model):
    def __init__(self):
        super(Darknet53, self).__init__(name='')
        
        # (256, 256, 3)
        self.l0a = _ConvBlock(32, layer_idx=0, name="stage0")
        self.l0_pool = _ConvPoolBlock(64, layer_idx=1, name="stage0")

        # (128, 128, 64)
        self.l1a = _ResidualBlock([32, 64], layer_idx=[2, 3], name="stage1")
        self.l1_pool = _ConvPoolBlock(128, layer_idx=4, name="stage1")

        # (64, 64, 128)
        self.l2a = _ResidualBlock([64, 128], layer_idx=[5, 6], name="stage2")
        self.l2b = _ResidualBlock([64, 128], layer_idx=[7, 8], name="stage2")
        self.l2_pool = _ConvPoolBlock(256, layer_idx=9, name="stage2")

        # (32, 32, 256)
        self.l3a = _ResidualBlock([128, 256], layer_idx=[10, 11], name="stage3")
        self.l3b = _ResidualBlock([128, 256], layer_idx=[12, 13], name="stage3")
        self.l3c = _ResidualBlock([128, 256], layer_idx=[14, 15], name="stage3")
        self.l3d = _ResidualBlock([128, 256], layer_idx=[16, 17], name="stage3")
        self.l3e = _ResidualBlock([128, 256], layer_idx=[18, 19], name="stage3")
        self.l3f = _ResidualBlock([128, 256], layer_idx=[20, 21], name="stage3")
        self.l3g = _ResidualBlock([128, 256], layer_idx=[22, 23], name="stage3")
        self.l3h = _ResidualBlock([128, 256], layer_idx=[24, 25], name="stage3")
        self.l3_pool = _ConvPoolBlock(512, layer_idx=26, name="stage3")
        
        # (16, 16, 512)
        self.l4a = _ResidualBlock([256, 512], layer_idx=[27, 28], name="stage4")
        self.l4b = _ResidualBlock([256, 512], layer_idx=[29, 30], name="stage4")
        self.l4c = _ResidualBlock([256, 512], layer_idx=[31, 32], name="stage4")
        self.l4d = _ResidualBlock([256, 512], layer_idx=[33, 34], name="stage4")
        self.l4e = _ResidualBlock([256, 512], layer_idx=[35, 36], name="stage4")
        self.l4f = _ResidualBlock([256, 512], layer_idx=[37, 38], name="stage4")
        self.l4g = _ResidualBlock([256, 512], layer_idx=[39, 40], name="stage4")
        self.l4h = _ResidualBlock([256, 512], layer_idx=[41, 42], name="stage4")
        self.l4_pool = _ConvPoolBlock(1024, layer_idx=43, name="stage4")

        # (8, 8, 1024)
        self.l5a = _ResidualBlock([512, 1024], layer_idx=[44, 45], name="stage5")
        self.l5b = _ResidualBlock([512, 1024], layer_idx=[46, 47], name="stage5")
        self.l5c = _ResidualBlock([512, 1024], layer_idx=[48, 49], name="stage5")
        self.l5d = _ResidualBlock([512, 1024], layer_idx=[50, 51], name="stage5")
        
        self.num_layers = 52
        self._init_vars()

    def call(self, input_tensor, training=False):
        
        x = self.l0a(input_tensor, training)
        x = self.l0_pool(x, training)

        x = self.l1a(x, training)
        x = self.l1_pool(x, training)

        x = self.l2a(x, training)
        x = self.l2b(x, training)
        x = self.l2_pool(x, training)

        x = self.l3a(x, training)
        x = self.l3b(x, training)
        x = self.l3c(x, training)
        x = self.l3d(x, training)
        x = self.l3e(x, training)
        x = self.l3f(x, training)
        x = self.l3g(x, training)
        x = self.l3h(x, training)
        output_stage3 = x
        x = self.l3_pool(x, training)

        x = self.l4a(x, training)
        x = self.l4b(x, training)
        x = self.l4c(x, training)
        x = self.l4d(x, training)
        x = self.l4e(x, training)
        x = self.l4f(x, training)
        x = self.l4g(x, training)
        x = self.l4h(x, training)
        output_stage4 = x
        x = self.l4_pool(x, training)

        x = self.l5a(x, training)
        x = self.l5b(x, training)
        x = self.l5c(x, training)
        x = self.l5d(x, training)
        output_stage5 = x
        return output_stage3, output_stage4, output_stage5
    
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
        import numpy as np
        imgs = np.random.randn(1, 256, 256, 3).astype(np.float32)
        input_tensor = tf.constant(imgs)
        self.call(input_tensor)


class _ConvBlock(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_ConvBlock, self).__init__(name=name)
        
        layer_name = "layer_{}".format(str(layer_idx))

        self.conv = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, name=layer_name)
        self.bn = layers.BatchNormalization(epsilon=0.001, name=layer_name)

    def call(self, input_tensor, training=False):

        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class _ConvPoolBlock(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_ConvPoolBlock, self).__init__(name=name)

        layer_name = "layer_{}".format(str(layer_idx))

        self.pad = layers.ZeroPadding2D(((1,0),(1,0)))
        self.conv = layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name=layer_name)
        self.bn = layers.BatchNormalization(epsilon=0.001, name=layer_name)

    def call(self, input_tensor, training=False):

        x = self.pad(input_tensor)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class _ResidualBlock(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_ResidualBlock, self).__init__(name=name)
        filters1, filters2 = filters
        layer1, layer2 = layer_idx

        layer_name1 = "layer_{}".format(str(layer1))
        layer_name2 = "layer_{}".format(str(layer2))

        self.conv2a = layers.Conv2D(filters1, (1, 1), padding='same', use_bias=False, name=layer_name1)
        self.bn2a = layers.BatchNormalization(epsilon=0.001, name=layer_name1)

        self.conv2b = layers.Conv2D(filters2, (3, 3), padding='same', use_bias=False, name=layer_name2)
        self.bn2b = layers.BatchNormalization(epsilon=0.001, name=layer_name2)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        x += input_tensor
        return x



