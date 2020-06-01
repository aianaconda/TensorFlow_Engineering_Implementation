"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class TextCNN(object):
    """
    TextCNN文本分类器.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        
        #定义占位符
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #词嵌入层 
        with tf.variable_scope('Embedding'):
            embed = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=vocab_size, embed_dim=embedding_size)
            self.embedded_chars_expanded = tf.expand_dims(embed, -1)

        #定义多通道卷积 与最大池化网络
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            conv = slim.conv2d(self.embedded_chars_expanded, num_outputs = num_filters, 
                            kernel_size=[filter_size,embedding_size], 
                            stride=1, padding="VALID",
                            activation_fn=tf.nn.leaky_relu,scope="conv%s" % filter_size)
            pooled = slim.max_pool2d(conv, [sequence_length - filter_size + 1, 1], padding='VALID',
                scope="pool%s" % filter_size)

            pooled_outputs.append(pooled)

        #展开特征，并添加dropout
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #计算L2_loss
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            self.scores = slim.fully_connected(self.h_drop, num_classes, activation_fn=None,scope="fully_connected" )
            for tf_var in tf.trainable_variables():
                if ("fully_connected" in tf_var.name ):
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
                    print("tf_var",tf_var)

            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 计算交叉熵
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        #计算准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
   
    def build_mode(self):#定义函数构建模型
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        #生成摘要
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        #生成损失及准确率的摘要
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        #合并摘要
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])

        