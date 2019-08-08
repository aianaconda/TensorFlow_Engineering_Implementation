# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import tensorflow as tf
model = __import__("5-2  model")
MyNASNetModel = model.MyNASNetModel

batch_size = 32
train_dir  = 'data/train'
val_dir  = 'data/val'

learning_rate1 = 1e-1
learning_rate2 = 1e-3

mymode = MyNASNetModel(r'nasnet-a_mobile_04_10_2017\model.ckpt')  #初始化模型
mymode.build_model('train',val_dir,train_dir,batch_size,learning_rate1 ,learning_rate2 )#将模型定义载入图中

num_epochs1 = 2   #微调的迭代次数
num_epochs2 = 200#联调的迭代次数

with tf.Session() as sess:
    sess.run(mymode.global_init)
   
    step = 0
    step = mymode.load_cpk(mymode.global_step,sess,1,mymode.saver,mymode.save_path )#载入模型
    print(step)
    if step == 0:#微调
        mymode.init_fn(sess)  #载入预编译模型权重
        
        for epoch in range(num_epochs1):
      		     
            print('Starting1 epoch %d / %d' % (epoch + 1, num_epochs1))#输出进度 
            #用训练集初始化迭代器
            sess.run(mymode.train_init_op)#数据集从头开始
            while True:
                try:
                    step += 1
                    #预测，合并图，训练
                    acc,accuracy_top_5, summary, _ = sess.run([mymode.accuracy, mymode.accuracy_top_5,mymode.merged,mymode.last_train_op])
                    
                    #mymode.train_writer.add_summary(summary, step)#写入日志文件
                    if step % 100 == 0:
                        print(f'step: {step} train1 accuracy: {acc},{accuracy_top_5}')
                except tf.errors.OutOfRangeError:#数据集指针在最后
                    print("train1:",epoch," ok")
                    mymode.saver.save(sess, mymode.save_path+"/mynasnet.cpkt",   global_step=mymode.global_step.eval())
                    break
    
        sess.run(mymode.step_init)#微调结束，计数器从0开始
    
    #整体训练
    for epoch in range(num_epochs2):
        print('Starting2 epoch %d / %d' % (epoch + 1, num_epochs2))
        sess.run(mymode.train_init_op)
        while True:
            try:
                step += 1
                #预测，合并图，训练
                acc, summary, _ = sess.run([mymode.accuracy, mymode.merged, mymode.full_train_op])
                
                mymode.train_writer.add_summary(summary, step)#写入日志文件

                if step % 100 == 0:
                    print(f'step: {step} train2 accuracy: {acc}')
            except tf.errors.OutOfRangeError:
                print("train2:",epoch," ok")
                mymode.saver.save(sess, mymode.save_path+"/mynasnet.cpkt",   global_step=mymode.global_step.eval())
                break

