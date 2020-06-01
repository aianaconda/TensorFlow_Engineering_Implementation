"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
tf.compat.v1.disable_v2_behavior()
class TextCNN(object):
    """
    TextCNN文本分类器.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        
        #定义占位符
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        #词嵌入层 
        embed_initer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
            
        embed = layers.Embedding(vocab_size, embedding_size,
                                  embeddings_initializer=embed_initer,
                                  input_length=sequence_length,
                                  name='Embedding')(self.input_x)
        
        embed = layers.Reshape((sequence_length, embedding_size, 1), name='add_channel')(embed)

        #定义多通道卷积 与最大池化网络
        pool_outputs = []
        
        for i, filter_size in enumerate(filter_sizes):            
            filter_shape = (filter_size, embedding_size)    
            
            conv = layers.Conv2D(num_filters, filter_shape, strides=(1, 1), padding='valid',
                                 activation=tf.nn.leaky_relu,
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=tf.keras.initializers.constant(0.1),
                                 name='convolution_{:d}'.format(filter_size))(embed)

            max_pool_shape = (sequence_length - filter_size + 1, 1)
            pool = layers.MaxPool2D(pool_size=max_pool_shape,
                                      strides=(1, 1), padding='valid',
                                      #data_format='channels_last',
                                      name='max_pooling_{:d}'.format(filter_size))(conv)            

            pool_outputs.append(pool)

        #展开特征，并添加dropout        
        pool_outputs = layers.concatenate(pool_outputs, axis=-1, name='concatenate')
        pool_outputs = layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
        pool_outputs = layers.Dropout(self.dropout_keep_prob, name='dropout')(pool_outputs)
        
        
        #计算L2_loss
        l2_loss = tf.constant(0.0)
        
        outputs = layers.Dense(num_classes, activation=None,
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=tf.keras.initializers.constant(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg_lambda),
                                 bias_regularizer=tf.keras.regularizers.l2(l2_reg_lambda),
                                 name='dense')(pool_outputs)
        
        for tf_var in tf.compat.v1.trainable_variables():
            if ("dense" in tf_var.name ):
                l2_loss += tf.reduce_mean(input_tensor=tf.nn.l2_loss(tf_var))
                print("tf_var",tf_var)

        self.predictions = tf.argmax(input=outputs, axis=1, name="predictions")

        # 计算交叉熵
        with tf.compat.v1.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=self.input_y)
            self.loss = tf.reduce_mean(input_tensor=losses) + l2_reg_lambda * l2_loss

        #计算准确率
        with tf.compat.v1.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(input=self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_predictions, "float"), name="accuracy")
   
    def build_mode(self):#定义函数构建模型
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        #生成摘要
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)
        #生成损失及准确率的摘要
        loss_summary = tf.compat.v1.summary.scalar("loss", self.loss)
        acc_summary = tf.compat.v1.summary.scalar("accuracy", self.accuracy)

        #合并摘要
        self.train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])

