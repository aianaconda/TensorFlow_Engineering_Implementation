# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
        
tf.reset_default_graph()
 
savedir = "log/"
PATH_TO_CKPT = savedir +'/expert-graph-yes.pb'

my_graph_def = tf.GraphDef() #定义GraphDef对象
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    my_graph_def.ParseFromString(serialized_graph)#读pb文件
    print(my_graph_def)
    tf.import_graph_def(my_graph_def, name='')#恢复到当前图中

my_graph = tf.get_default_graph()  #获得当前图
result = my_graph.get_tensor_by_name('add:0')#获得当前图中的z赋值给result
x = my_graph.get_tensor_by_name('Placeholder:0')#获得当前图中的X赋值给x    
        
with tf.Session() as sess:
    y = sess.run(result, feed_dict={x: 5})#传入5，进行预测
    print(y)
        

       