# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class CapsuleNetModel:  #定义胶囊网络模型类
    def __init__(self, batch_size,n_classes,iter_routing):#初始化
        self.batch_size=batch_size
        self.n_classes = n_classes
        self.iter_routing = iter_routing

    def CapsuleNet(self, img):#定义网络模型结构
        
        with tf.variable_scope('Conv1_layer') as scope:#定义第一个正常卷积层 ReLU Conv1
            output = slim.conv2d(img, num_outputs=256, kernel_size=[9, 9], stride=1, padding='VALID', scope=scope)
            assert output.get_shape() == [self.batch_size, 20, 20, 256]
    
        with tf.variable_scope('PrimaryCaps_layer') as scope:#定义主胶囊网络 Primary Caps
            output = slim.conv2d(output, num_outputs=32*8, kernel_size=[9, 9], stride=2, padding='VALID', scope=scope, activation_fn=None)
            output = tf.reshape(output, [self.batch_size, -1, 1, 8])  #将结果变成32*6*6个胶囊单元，每个单元为8维向量
            assert output.get_shape() == [self.batch_size, 1152, 1, 8]

        with tf.variable_scope('DigitCaps_layer') as scope:#定义数字胶囊 Digit Caps
            u_hats = []
            input_groups = tf.split(axis=1, num_or_size_splits=1152, value=output)#将输入按照胶囊单元分开
            for i in range(1152): #遍历每个胶囊单元
                #利用卷积核为[1，1]的卷积操作，让u与w相乘，再相加得到u_hat
                one_u_hat = slim.conv2d(input_groups[i], num_outputs=16*10, kernel_size=[1, 1], stride=1, padding='VALID', scope='DigitCaps_layer_w_'+str(i), activation_fn=None)
                one_u_hat = tf.reshape(one_u_hat, [self.batch_size, 1, 10, 16])#每个胶囊单元变成了16维向量
                u_hats.append(one_u_hat)
            
            u_hat = tf.concat(u_hats, axis=1)#将所有的胶囊单元中的one_u_hat合并起来
            assert u_hat.get_shape() == [self.batch_size, 1152, 10, 16]

            #初始化b值
            b_ijs = tf.constant(np.zeros([1152, 10], dtype=np.float32))
            v_js = []
            for r_iter in range(self.iter_routing):#按照指定循环次数，计算动态路由
                with tf.variable_scope('iter_'+str(r_iter)):
                    c_ijs = tf.nn.softmax(b_ijs, axis=1)  #根据b值，获得耦合系数

                    #将下列变量按照10类分割，每一类单独运算
                    c_ij_groups = tf.split(axis=1, num_or_size_splits=10, value=c_ijs)
                    b_ij_groups = tf.split(axis=1, num_or_size_splits=10, value=b_ijs)
                    u_hat_groups = tf.split(axis=2, num_or_size_splits=10, value=u_hat)

                    for i in range(10):
                        #生成具有跟输入一样尺寸的卷积核[1152, 1]，输入为16通道,卷积核个数为1个
                        c_ij = tf.reshape(tf.tile(c_ij_groups[i], [1, 16]), [1152, 1, 16, 1])
                        #利用深度卷积实现u_hat与c矩阵的对应位置相乘，输出的通道数为16*1个
                        s_j = tf.nn.depthwise_conv2d(u_hat_groups[i], c_ij, strides=[1, 1, 1, 1], padding='VALID')
                        assert s_j.get_shape() == [self.batch_size, 1, 1, 16]

                        s_j = tf.reshape(s_j, [self.batch_size, 16])
                        v_j = self.squash(s_j)  #使用squash激活函数，生成最终的输出vj
                        assert v_j.get_shape() == [self.batch_size, 16]
                        #根据vj来计算，并更新b值
                        b_ij_groups[i] = b_ij_groups[i]+tf.reduce_sum(tf.matmul(tf.reshape(u_hat_groups[i], 
                                   [self.batch_size, 1152, 16]), tf.reshape(v_j, [self.batch_size, 16, 1])), axis=0)

                        if r_iter == self.iter_routing-1:  #迭代结束后，再生成一次vj，得到数字胶囊真正的输出结果
                            v_js.append(tf.reshape(v_j, [self.batch_size, 1, 16]))

                    b_ijs = tf.concat(b_ij_groups, axis=1)#将10类的b合并到一起

            output = tf.concat(v_js, axis=1)#将10类的vj合并到一起，生成的形状为[self.batch_size, 10, 16]的结果

        return  output
    def squash(self, s_j):  #定义激活函数
        s_j_norm_square = tf.reduce_mean(tf.square(s_j), axis=1, keepdims=True)
        v_j = s_j_norm_square*s_j/((1+s_j_norm_square)*tf.sqrt(s_j_norm_square+1e-9))
        return v_j
    
    def build_model(self, is_train=False,learning_rate = 1e-3):
        tf.reset_default_graph()

        #定义占位符
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.n_classes])
        self.x = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='input')
        
        #定义计步器
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        biasInitializer = tf.constant_initializer(0.0)
    
        with slim.arg_scope([slim.conv2d], trainable=is_train, weights_initializer=initializer, biases_initializer=biasInitializer):
            self.v_jsoutput = self.CapsuleNet(self.x) #构建胶囊网络
            
            tf.check_numerics(self.v_jsoutput,"self.v_jsoutput is nan ")#判断张量是否为nan 或inf
            
            with tf.variable_scope('Masking'):  
                self.v_len = tf.norm(self.v_jsoutput, axis=2)#计算输出值的欧几里得范数[self.batch_size, 10]
                
                
    
            if is_train:            #如果是训练模式，重建输入图片
                masked_v = tf.matmul(self.v_jsoutput, tf.reshape(self.y, [-1, 10, 1]), transpose_a=True)
                masked_v = tf.reshape(masked_v, [-1, 16])
    
                with tf.variable_scope('Decoder'):
                    output = slim.fully_connected(masked_v, 512, trainable=is_train)
                    output = slim.fully_connected(output, 1024, trainable=is_train)
                    self.output = slim.fully_connected(output, 784, trainable=is_train, activation_fn=tf.sigmoid)
    
                self.total_loss = self.loss(self.v_len,self.output)#计算loss值
                #使用退化学习率
                learning_rate_decay = tf.train.exponential_decay(learning_rate, global_step=self.global_step, decay_steps=2000,decay_rate=0.9)
    
                #定义优化器
                self.train_op = tf.train.AdamOptimizer(learning_rate_decay).minimize(self.total_loss, global_step=self.global_step)
                
        #定义保存及恢复模型关键点所使用的saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    def loss(self,v_len, output): #定义loss计算函数
        max_l = tf.square(tf.maximum(0., 0.9-v_len))
        max_r = tf.square(tf.maximum(0., v_len - 0.1))
        
        l_c = self.y*max_l+0.5 * (1 - self.y) * max_r
    
        margin_loss = tf.reduce_mean(tf.reduce_sum(l_c, axis=1))
    
        origin = tf.reshape(self.x, shape=[self.batch_size, -1])
        reconstruction_err = tf.reduce_mean(tf.square(output-origin))
    
        total_loss = margin_loss+0.0005*reconstruction_err#将边距损失与重建损失一起构成loss

    
        return total_loss

