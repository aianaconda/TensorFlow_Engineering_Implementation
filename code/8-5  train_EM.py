"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os
import numpy as np


import sys

capsnet_em = __import__("8-4  capsnet_em")
build_em = capsnet_em.build_em
spread_loss= capsnet_em.spread_loss
test_accuracy= capsnet_em.test_accuracy
#载入数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./fashion/", one_hot=False)


mnist.train.num_examples

num_classes = 10
tf.logging.set_verbosity(tf.logging.INFO)


tf.reset_default_graph()

batch_feature = tf.placeholder(tf.float32, [None,  28, 28, 1])
batch_labels = tf.placeholder(tf.int32, [None, ])



decay_epoch = 1000

logger =tf.logging
batch_size =64

m_min = 0.2
m_max = 0.9
m = m_min

with tf.device('/gpu:0'):
     
    with slim.arg_scope([slim.variable], device='/cpu:0'):
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        """Define the dataflow graph."""
        m_op = tf.placeholder(dtype=tf.float32, shape=())

        batch_x_squash = tf.divide(batch_feature, 255.)
        batch_x = slim.batch_norm(batch_feature, center=False, is_training=True, trainable=True)
        
        output, pose_out = build_em(batch_x,batch_size, is_train=True,num_classes=num_classes)
        print(pose_out.get_shape())
        loss, _, _, _ = spread_loss( output,batch_size, pose_out, 
                                                    batch_x_squash, batch_labels, m)
        modeldir = './model3'
        acc,logits_idx = test_accuracy(output, batch_labels)
        
    lrn_rate = tf.maximum(tf.train.exponential_decay( 1e-3, global_step, decay_epoch, 0.8), 1e-5)
    opt = tf.train.AdamOptimizer()  

    grad = opt.compute_gradients(loss)
    grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                  for g, _ in grad if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]


with tf.control_dependencies(grad_check):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grad, global_step=global_step)
        
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

training_epochs  = 50
ckpt_path = os.path.join(modeldir, 'model.ckpt')
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                      log_device_placement=False)) as sess:  #建立会话
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    checkpoint_path = tf.train.latest_checkpoint(modeldir)#载入检查点
    print("checkpoint_path",checkpoint_path)
    if checkpoint_path !=None:
        saver.restore(sess, checkpoint_path)
       
    for epoch in range(training_epochs):#按照指定次数迭代数据集
        total_batch = int(mnist.train.num_examples/batch_size)
        for step in range(total_batch):  #遍历数据集
            data_x, data_y = mnist.train.next_batch(batch_size)#取出数据
            data_x = np.reshape(data_x,[batch_size, 28, 28, 1])
            tic = time.time()  
           
            try: 
                _, loss_value = sess.run( [train_op, loss],
                                         feed_dict={batch_feature: data_x, batch_labels : data_y} )

                if step % 20 == 0:#每训练20次，输出一次结果
                    accv = sess.run( acc, feed_dict={batch_feature: data_x, batch_labels : data_y} )
                    print(epoch,'epoch ，%d iteration finishs in ' % step + '%f second' %
                                (time.time() - tic) + ' loss=%f' % loss_value,"acc",accv)
                    
                    #saver.save(sess, ckpt_path, global_step=step)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                saver.save(sess, ckpt_path, global_step=step)
                sys.exit()
            except tf.errors.InvalidArgumentError:
                print('%d iteration contains NaN gradients. Discard.' % step)
                continue

        saver.save(sess, ckpt_path, global_step=step)



