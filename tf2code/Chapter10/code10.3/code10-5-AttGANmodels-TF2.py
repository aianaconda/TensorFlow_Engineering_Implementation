"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import tensorflow.keras.layers as KL
#import tensorflow.contrib.slim as slim

MAX_DIM = 64 * 16
def Genc(x, dim=64, n_layers=5, is_training=True):
    with tf.compat.v1.variable_scope('Genc', reuse=tf.compat.v1.AUTO_REUSE):
        z = x
        zs = []
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            z=KL.Conv2D(d,4,2,activation=tf.nn.leaky_relu)(z)
            #z = slim.conv2d(z,d,4,2,activation_fn=tf.nn.leaky_relu)
            #z = slim.batch_norm(z,scale=True,updates_collections=None, is_training=is_training)
            z = KL.BatchNormalization(trainable=is_training)(z)
            zs.append(z)
        return zs


def Gdec(zs, _a, dim=64, n_layers=5, shortcut_layers=1, inject_layers=0, is_training=True):
    shortcut_layers = min(shortcut_layers, n_layers - 1)
    inject_layers = min(inject_layers, n_layers - 1)

    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            _a = tf.reshape(_a, [-1, 1, 1, _a.get_shape()[-1] ])
            _a = tf.tile(_a, [1, z.get_shape()[1],z.get_shape()[2], 1])
            feats.append(_a)
        return tf.concat(feats, axis=3)

    with tf.compat.v1.variable_scope('Gdec', reuse=tf.compat.v1.AUTO_REUSE):
        z = _concat(zs[-1], None, _a)#z.shape is (32, 2, 2, 1037) 
        for i in range(n_layers):
            if i < n_layers - 1:
                d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
                #z = slim.conv2d_transpose(z,d,4,2,activation_fn=tf.nn.relu)
                z = KL.Conv2DTranspose(d,4,2,activation=tf.nn.relu)(z)#z.shape is (32, 6, 6, 1037)
                #z = slim.batch_norm(z,scale=True,updates_collections=None, is_training=is_training)
                z = KL.BatchNormalization(trainable=is_training)(z)#z.shape is (32, 6, 6, 1037)
                if shortcut_layers > i:
                    z = _concat(z, zs[n_layers - 2 - i], None)
                if inject_layers > i:
                    z = _concat(z, None, _a)
            else:
                #x = slim.conv2d_transpose(z, 3, 4, 2,activation_fn=tf.nn.tanh)
                x = KL.Conv2DTranspose(3, 6, 2,activation=tf.nn.tanh)(z)#(32,126,126,3)
        return x


def D(x, n_att, dim=64, fc_dim=MAX_DIM, n_layers=5):
    with tf.compat.v1.variable_scope('D', reuse=tf.compat.v1.AUTO_REUSE):
        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            #y= slim.conv2d(y,d,4,2, normalizer_fn=slim.instance_norm,activation_fn=tf.nn.leaky_relu)
            y= KL.Conv2D(d,4,2, activation=tf.nn.leaky_relu)(y)
        print(y.shape,y.shape.ndims)
        if y.shape.ndims > 2:#大于2维，需要展开。变成2维的再做全连接
            #y = KL.Flatten(y)
            y=tf.compat.v1.layers.flatten(y)
        #logit_gan = slim.fully_connected(y, fc_dim,activation_fn =tf.nn.leaky_relu )
        logit_gan = KL.Dense(fc_dim,activation =tf.nn.leaky_relu )(y)
        #logit_gan = slim.fully_connected(logit_gan, 1,activation_fn =None )
        logit_gan = KL.Dense(1,activation =None )(logit_gan)
        #logit_att = slim.fully_connected(y, fc_dim,activation_fn =tf.nn.leaky_relu )
        logit_att = KL.Dense(fc_dim,activation =tf.nn.leaky_relu )(y)
        #logit_att = slim.fully_connected(logit_att, n_att,activation_fn =None )
        logit_att = KL.Dense(n_att,activation =None )(logit_att)

        return logit_gan, logit_att


def gradient_penalty(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.compat.v1.name_scope('interpolate'):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(input=a), minval=0., maxval=1.)
                _, variance = tf.nn.moments(x=a, axes=range(a.shape.ndims))
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(input=a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.compat.v1.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(ys=pred, xs=x)[0]
        #norm = tf.norm(tensor=slim.flatten(grad), axis=1)
        norm = tf.norm(tensor=tf.compat.v1.layers.flatten(grad), axis=1)
        gp = tf.reduce_mean(input_tensor=(norm - 1.)**2)
        return gp
