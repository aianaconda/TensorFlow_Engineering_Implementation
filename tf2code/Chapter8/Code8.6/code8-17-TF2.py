# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


# Darknet53 feature extractor
class Headnet(tf.keras.Model):
    def __init__(self, n_classes=80):
        super(Headnet, self).__init__(name='')
        n_features = 3 * (n_classes+1+4)
        
        self.stage5_conv5 = _Conv5([512, 1024, 512, 1024, 512],
                                   [75, 76, 77, 78, 79])
        self.stage5_conv2 = _Conv2([1024, n_features],
                                   [80, 81],
                                   "detection_layer_1_{}".format(n_features))
        self.stage5_upsampling = _Upsamling([256], [84])

        self.stage4_conv5 = _Conv5([256, 512, 256, 512, 256],
                                   [87, 88, 89, 90, 91])
        self.stage4_conv2 = _Conv2([512, n_features],
                                   [92, 93],
                                   "detection_layer_2_{}".format(n_features))
        self.stage4_upsampling = _Upsamling([128], [96])

        self.stage3_conv5 = _Conv5([128, 256, 128, 256, 128],
                                   [99, 100, 101, 102, 103])
        self.stage3_conv2 = _Conv2([256, n_features],
                                   [104, 105],
                                   "detection_layer_3_{}".format(n_features))
        self.num_layers = 106
        self._init_vars()

    def call(self, stage3_in, stage4_in, stage5_in, training=False):
        x = self.stage5_conv5(stage5_in, training)
        stage5_output = self.stage5_conv2(x, training)

        x = self.stage5_upsampling(x, training)
        x = layers.concatenate([x, stage4_in])
        x = self.stage4_conv5(x, training)
        stage4_output = self.stage4_conv2(x, training)

        x = self.stage4_upsampling(x, training)
        x = layers.concatenate([x, stage3_in])
        x = self.stage3_conv5(x, training)
        stage3_output = self.stage3_conv2(x, training)
        return stage5_output, stage4_output, stage3_output

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
        s3 = tf.constant(np.random.randn(1, 32, 32, 256).astype(np.float32))
        s4 = tf.constant(np.random.randn(1, 16, 16, 512).astype(np.float32))
        s5 = tf.constant(np.random.randn(1, 8, 8, 1024).astype(np.float32))
        self.call(s3, s4, s5, training=False)


class _Conv5(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_Conv5, self).__init__(name=name)
        
        layer_names = ["layer_{}".format(i) for i in layer_idx]

        self.conv1 = layers.Conv2D(filters[0], (1, 1), strides=(1, 1), padding='same', use_bias=False, name=layer_names[0])
        self.bn1 = layers.BatchNormalization(epsilon=0.001, name=layer_names[0])

        self.conv2 = layers.Conv2D(filters[1], (3, 3), strides=(1, 1), padding='same', use_bias=False, name=layer_names[1])
        self.bn2 = layers.BatchNormalization(epsilon=0.001, name=layer_names[1])

        self.conv3 = layers.Conv2D(filters[2], (1, 1), strides=(1, 1), padding='same', use_bias=False, name=layer_names[2])
        self.bn3 = layers.BatchNormalization(epsilon=0.001, name=layer_names[2])

        self.conv4 = layers.Conv2D(filters[3], (3, 3), strides=(1, 1), padding='same', use_bias=False, name=layer_names[3])
        self.bn4 = layers.BatchNormalization(epsilon=0.001, name=layer_names[3])

        self.conv5 = layers.Conv2D(filters[4], (1, 1), strides=(1, 1), padding='same', use_bias=False, name=layer_names[4])
        self.bn5 = layers.BatchNormalization(epsilon=0.001, name=layer_names[4])


    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class _Conv2(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_Conv2, self).__init__(name=name)
        
        layer_names = ["layer_{}".format(i) for i in layer_idx]

        self.conv1 = layers.Conv2D(filters[0], (3, 3), strides=(1, 1), padding='same', use_bias=False, name=layer_names[0])
        self.bn = layers.BatchNormalization(epsilon=0.001, name=layer_names[0])
        self.conv2 = layers.Conv2D(filters[1], (1, 1), strides=(1, 1), padding='same', use_bias=True, name=layer_names[1])

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv2(x)
        return x


class _Upsamling(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_Upsamling, self).__init__(name=name)
        
        layer_names = ["layer_{}".format(i) for i in layer_idx]

        self.conv = layers.Conv2D(filters[0], (1, 1), strides=(1, 1), padding='same', use_bias=False, name=layer_names[0])
        self.bn = layers.BatchNormalization(epsilon=0.001, name=layer_names[0])
        self.upsampling = layers.UpSampling2D(2)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.upsampling(x)
        return x


if __name__ == '__main__':
    import numpy as np
    s3 = tf.constant(np.random.randn(1, 32, 32, 256).astype(np.float32))
    s4 = tf.constant(np.random.randn(1, 16, 16, 512).astype(np.float32))
    s5 = tf.constant(np.random.randn(1, 8, 8, 1024).astype(np.float32))
    
    # (1, 256, 256, 3) => (1, 8, 8, 1024)
    headnet = Headnet()
    f5, f4, f3 = headnet(s3, s4, s5)
    print(f5.shape, f4.shape, f3.shape)

    for v in headnet.variables:
        print(v.name)



