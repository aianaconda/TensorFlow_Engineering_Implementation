# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import os
import time
import datetime

predata = __import__("8-6  NLP文本预处理")
mydataset = predata.mydataset
text_cnn = __import__("8-7  TextCnn模型")
TextCNN = text_cnn.TextCNN
    
def train():
    #指定样本文件
    positive_data_file ="./data/rt-polaritydata/rt-polarity.pos"
    negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"
    #设置训练参数
    num_steps = 2000            #定义训练次数
    display_every=20            #定义训练中的显示间隔
    checkpoint_every=100        #定义训练中保存模型的间隔
    SaveFileName= "text_cnn_model" #定义保存模型文件夹名称
    #设置模型参数    
    num_classes =2          #设置模型分类
    dropout_keep_prob =0.8  #定义dropout系数
    l2_reg_lambda=0.1       #定义正则化系数
    filter_sizes = "3,4,5"  #定义多通道卷积核
    num_filters =64         #定义每通道的输出个数
    
    tf.reset_default_graph()#清空图
    
    #预处理生成字典及数据集
    data,vocab_processor,max_document_length =mydataset(positive_data_file,negative_data_file)
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    #定义TextCnn网络
    cnn = TextCNN(
        sequence_length=max_document_length,
        num_classes=num_classes,
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=128,
        filter_sizes=list(map(int, filter_sizes.split(","))),
        num_filters=num_filters,
        l2_reg_lambda=l2_reg_lambda)
    #构建网络
    cnn.build_mode()

    #打开session，准备训练
    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        

        #准备输出模型路径
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, SaveFileName, timestamp))
        print("Writing to {}\n".format(out_dir))

        #准备输出摘要路径
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        

        #准备检查点名称
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #定义保存检查点的saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        #保存字典
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        def train_step(x_batch, y_batch):#训练步骤
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [cnn.train_op, cnn.global_step, cnn.train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            train_summary_writer.add_summary(summaries, step)
            return (time_str, step, loss, accuracy)

        i = 0
        while  tf.train.global_step(sess, cnn.global_step) < num_steps:
            x_batch, y_batch = sess.run(next_element)
            i = i+1
            time_str, step, loss, accuracy =train_step(x_batch, y_batch)
            
            current_step = tf.train.global_step(sess, cnn.global_step)
            if current_step % display_every == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    train()#启动训练

if __name__ == '__main__':
    tf.app.run()