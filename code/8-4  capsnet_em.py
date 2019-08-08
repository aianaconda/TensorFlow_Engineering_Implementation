"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

weight_reg = False  #是否使用参数正则化
epsilon=1e-9#防止分母为0的最小数
iter_routing=2  #EM算法的迭代次数

def build_em(input, batch_size,is_train: bool, num_classes: int):
    A,B,C,D=32,8,16,16 #定义各层的输出维度
    
    data_size = int(input.get_shape()[1])#输入尺寸为28*28
    bias_initializer = tf.truncated_normal_initializer( mean=0.0, stddev=0.01)
    #参数l2正则层
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    tf.logging.info('input shape: {}'.format(input.get_shape()))#(?, 28, 28, 1)

    #为卷积权重统一初始化
    with slim.arg_scope([slim.conv2d], trainable=is_train, biases_initializer=bias_initializer, weights_regularizer=weights_regularizer):
        with tf.variable_scope('relu_conv1') as scope:   #relu_conv1层
            output = slim.conv2d(input, num_outputs=A, kernel_size=[
                                 5, 5], stride=2, padding='VALID', scope=scope, activation_fn=tf.nn.relu)
            data_size = int(np.floor((data_size - 5+1) / 2))#计算卷积后的尺寸，得到12
            tf.logging.info('conv1 output shape: {}'.format(output.get_shape()))#输出(?, 12, 12, 32)

        with tf.variable_scope('primary_caps') as scope:     #primary_caps层
            pose = slim.conv2d(output, num_outputs=B * 16,   #计算姿态矩阵
                               kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            pose = tf.reshape(pose, [batch_size, data_size, data_size, B, 16])
            
            activation = slim.conv2d(output, num_outputs=B, kernel_size=[  #计算激活值
                                     1, 1], stride=1, padding='VALID', scope='primary_caps/activation', activation_fn=tf.nn.sigmoid)
            activation = tf.reshape(activation, [batch_size, data_size, data_size, B, 1])
            
            output = tf.concat([pose, activation], axis=4)                 #计算primary_caps层输出
            output = tf.reshape(output, shape=[batch_size, data_size, data_size, -1])
            assert output.get_shape()[1:] == [ data_size, data_size, B * 17]
            tf.logging.info('primary capsule output shape: {}'.format(output.get_shape()))#(batch_size, 12, 12, 136)

        with tf.variable_scope('conv_caps1') as scope: #conv_caps1层
            pose ,activation = conv_caps(output,3,2,C,weights_regularizer,"conv cap 1")
            data_size = pose.get_shape()[1]
            
            #生成conv_caps1层结果
            output = tf.reshape(tf.concat([pose, activation], axis=4), [
                                batch_size, data_size, data_size, C*17])
            tf.logging.info('conv cap 1 output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('conv_caps2') as scope:
            pose ,activation = conv_caps(output,3,1,D,weights_regularizer,"conv cap 2")
            data_size = activation.get_shape()[1]

        with tf.variable_scope('class_caps') as scope:
            pose = tf.reshape(pose, [-1, D, 16])#调整形状
            activation = tf.reshape(activation, [-1, D, 1])
            #计算EM，获得姿态矩阵和激活值
            miu,activation = calEM(pose,activation,num_classes,weights_regularizer,"class cap")
            #调整形状
            activation = tf.reshape(activation, [ batch_size, data_size, data_size, num_classes])
            miu = tf.reshape(miu, [batch_size, data_size, data_size, -1])#(64, 3, 3, 16, 1)
            tf.logging.info('class caps activation: {}'.format(activation.get_shape()))
            tf.logging.info('class caps miu: {}'.format(miu.get_shape()))

        output = tf.nn.avg_pool(activation, ksize=[1, data_size, data_size, 1], strides=[
                            1, 1, 1, 1], padding='VALID')
        #最终分类结果
        output = tf.reshape(output,[batch_size, num_classes])
        tf.logging.info('class caps : {}'.format(output.get_shape()))

        pose = tf.nn.avg_pool(miu, ksize=[ #获得每一类的最终特征
                      1, data_size, data_size, 1], strides=[1, 1, 1, 1], padding='VALID')
        pose_out = tf.reshape(pose, shape=[batch_size, num_classes, 16])

    return output, pose_out

def conv_caps(indata,kernel,stride,outputdim,weights_regularizer,name):
    batch_size =int( indata.get_shape()[0])
    data_size = int(indata.get_shape()[1]) #尺寸（默认h和w相等）
    output = kernel_tile(indata, kernel, stride)#将主胶囊层的输出分成9个特征
    data_size = int(np.floor((data_size - kernel+1) / stride))#计算卷积后的尺寸

            
    newbatch = batch_size * data_size * data_size
    
    output = tf.reshape(output, shape=[newbatch, -1, 17])#[newbatch,kernel * kernel * 上层维度, 17]
    activation = tf.reshape(output[:, :, 16], [newbatch, -1, 1])
        
    miu,activation = calEM(output[:, :, :16],activation,outputdim,weights_regularizer,name)

    #生成姿态矩阵
    pose = tf.reshape(miu, [batch_size, data_size, data_size, outputdim, 16])
    tf.logging.info('{} pose shape: {}'.format(name,pose.get_shape()))
    #生成激活
    activation = tf.reshape(activation, [batch_size, data_size, data_size,outputdim, 1])
    tf.logging.info('{} activation shape: {}'.format(name,activation.get_shape()))
    
    return pose, activation


def kernel_tile(input, kernel, stride):

    input_shape = input.get_shape()
    #定义卷积核，输入ch为input_shape[3]，卷积核个数（ch）为kernel * kernel
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3], 
                                  kernel * kernel], dtype=np.float32)
    #为这9个卷积核赋值，每个卷积核的3*3矩阵中，有一个为1
    for i in range(kernel):  
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0   #kernel=3 ，步长为2，可以理解成分成9个一段。从中取样一个
                                                        
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    #深度卷积，在12*12上，按照3*3进行卷积。由于每个卷积核只有一个1，相当于采样
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[ 
                                    1, stride, stride, 1], padding='VALID')
    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[int(output_shape[0]), int(output_shape[1]), 
                                       int(output_shape[2]), int(input_shape[3]), kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3]) #（batch，5，5，9，ch）

    return output

def calEM(pose,activation,votes_output,weights_regularizer,name):
    with tf.variable_scope('v') as scope:#计算投票
        votes = mat_transform(pose, votes_output, weights_regularizer)        
        tf.logging.info('{} votes shape: {}'.format(name,votes.get_shape()))#(576, 16, 10, 16)

    with tf.variable_scope('routing') as scope2:#计算em路由，得到最终的姿态矩阵和激活值
        miu, activation= em_routing(votes, activation, votes_output, weights_regularizer)
        tf.logging.info(
            '{} activation shape: {}'.format(name,activation.get_shape()))
    return miu, activation



#定义一组权重，依次与输入中批次中的每个矩阵元素相乘。得到投票。相当于通过权重相乘的方式，又扩大了caps_num_c倍个姿态矩阵
#输入为[batch_size, caps_num_i, 16]--输出为[batch_size, caps_num_i,caps_num_c, 16]
def mat_transform(input, caps_num_c, regularizer, tag=False):
    batch_size = int(input.get_shape()[0])
    caps_num_i = int(input.get_shape()[1])#3 * 3 * B
    output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])

    w = slim.variable('w', shape=[1, caps_num_i, caps_num_c, 4, 4], dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                      regularizer=regularizer)

    w = tf.tile(w, [batch_size, 1, 1, 1, 1])#使用tile代替循环相乘。提升效率
    output = tf.tile(output, [1, 1, caps_num_c, 1, 1])
    #votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])
    votes = tf.reshape(output@w, [batch_size, caps_num_i, caps_num_c, 16])

    return votes



ac_lambda0=0.01

def em_routing(votes, activation, caps_num_c, regularizer, tag=False):

    batch_size = int(votes.get_shape()[0])
    caps_num_i = int(activation.get_shape()[1])
    n_channels = int(votes.get_shape()[-1])#姿态矩阵 16
    print("n_channels",n_channels)

    sigma_square = []
    miu = []
    activation_out = []
    #caps_num_c个投票，每个投票n_channels+激活值
    beta_v = slim.variable('beta_v', shape=[caps_num_c, n_channels], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),
                           regularizer=regularizer)
    beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),
                           regularizer=regularizer)

    votes_in = votes
    activation_in = activation

    for iters in range(iter_routing):
        # if iters == cfg.iter_routing-1:

        # e-step
        if iters == 0:#第一次，caps_num_c中的每个概率都一样.与16个父胶囊（分类）之间的连接
            r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
        else:
            log_p_c_h = -tf.log( tf.sqrt(sigma_square)) - (tf.square(votes_in - miu) / (2 * sigma_square)  )
            log_p_c_h = log_p_c_h - \
                        (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
                        
            p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

            ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

            r = ap / (tf.reduce_sum(ap, axis=2, keepdims=True) + epsilon)

        # m-step
        r = r * activation_in #更新概率=在原有的激活值基础上在乘属于该类的概率
        r = r / (tf.reduce_sum(r, axis=2, keepdims=True)+epsilon)#将数值转化为总数的占比（总数为1）

        r_sum = tf.reduce_sum(r, axis=1, keepdims=True)#所有胶囊的父胶囊连接概率收集起来。
        r1 = tf.reshape(r / (r_sum + epsilon),
                        shape=[batch_size, caps_num_i, caps_num_c, 1])

        miu = tf.reduce_sum(votes_in * r1, axis=1, keepdims=True)
        sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                     axis=1, keepdims=True) + epsilon

        if iters == iter_routing-1:
            r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
            cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,#计算信息熵
                                                         shape=[batch_size, caps_num_c, n_channels])))) * r_sum

            activation_out = tf.nn.softmax(ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
        else:
            activation_out = tf.nn.softmax(r_sum)

    return miu, activation_out
def spread_loss(output, batch_size,pose_out, x, y, m):

    num_class = int(output.get_shape()[-1])
    data_size = int(x.get_shape()[1])

    y = tf.one_hot(y, num_class, dtype=tf.float32)

    output1 = tf.reshape(output, shape=[tf.shape(y)[0], 1, num_class])
    y = tf.expand_dims(y, axis=2)
    #at = tf.matmul(output1, y)
    at = output1@y

    loss = tf.square(tf.maximum(0., m - (at - output1)))
    #loss = tf.matmul(loss, 1. - y)
    loss = loss@( 1. - y)
    loss = tf.reduce_mean(loss)
    
    print(pose_out.get_shape(),y.get_shape())

    pose_out = tf.reshape(tf.multiply(pose_out, y), shape=[batch_size, -1])
    tf.logging.info("decoder input value dimension:{}".format(pose_out.get_shape()))

    with tf.variable_scope('decoder'):
        pose_out = slim.fully_connected(pose_out, 512, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        pose_out = slim.fully_connected(pose_out, 1024, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        pose_out = slim.fully_connected(pose_out, data_size * data_size,
                                        trainable=True, activation_fn=tf.sigmoid, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))

        x = tf.reshape(x, shape=[tf.shape(y)[0], -1])
        reconstruction_loss = tf.reduce_mean(tf.square(pose_out - x))

    if weight_reg:
        # regularization loss
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss+0.0005*reconstruction_loss+regularization#
        loss_all = tf.add_n([loss] + [0.0005 * data_size* data_size * reconstruction_loss] + regularization)
    else:
        loss_all = tf.add_n([loss] + [0.0005 * data_size* data_size * reconstruction_loss])

    return loss_all, loss, reconstruction_loss, pose_out




def test_accuracy(logits, labels):
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(tf.shape(labels)[0],))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.shape(labels)[0], tf.float32)

    return accuracy,logits_idx