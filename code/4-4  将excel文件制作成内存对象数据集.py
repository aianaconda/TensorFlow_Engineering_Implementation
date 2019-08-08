# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import tensorflow as tf

def read_data(file_queue):                          #csv文件处理函数
    reader = tf.TextLineReader(skip_header_lines=1)  #tf.TextLineReader 可以每次读取一行
    key, value = reader.read(file_queue)
    
    defaults = [[0], [0.], [0.], [0.], [0.], [0]]       #为每个字段设置初始值
    cvscolumn = tf.decode_csv(value, defaults)           #tf.decode_csv对每一行进行解析
    
    featurecolumn = [i for i in cvscolumn[1:-1]]        #分离出列中的样本数据列
    labelcolumn = cvscolumn[-1]                         #分离出列中的标签数据列
    
    return tf.stack(featurecolumn), labelcolumn         #返回结果

def create_pipeline(filename, batch_size, num_epochs=None): #创建队列数据集函数
    #创建一个输入队列
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    
    feature, label = read_data(file_queue)              #载入数据和标签
    
    min_after_dequeue = 1000                            #设置队列中的最少数据条数（取完数据后，保证还是有1000条）
    capacity = min_after_dequeue + batch_size              #队列的长度
    
    feature_batch, label_batch = tf.train.shuffle_batch(    #生成乱序的批次数据
        [feature, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return feature_batch, label_batch                   #返回指定批次数据

#读取训练集
x_train_batch, y_train_batch = create_pipeline('iris_training.csv', 32, num_epochs=100)
#读取测试集
x_test, y_test = create_pipeline('iris_test.csv', 32)

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()                 #初始化
    local_init_op = tf.local_variables_initializer()            #初始化本地变量，没有回报错
    sess.run(init_op)
    sess.run(local_init_op)

    coord = tf.train.Coordinator()                          #创建协调器
    threads = tf.train.start_queue_runners(coord=coord)    #开启线程列队

    try:
        while True:
            if coord.should_stop():
                break
            example, label = sess.run([x_train_batch, y_train_batch]) #注入训练数据
            print ("训练数据：",example) #打印数据
            print ("训练标签：",label) #打印标签
    except tf.errors.OutOfRangeError:       #定义取完数据的异常处理
        print ('Done reading')
        example, label = sess.run([x_test, y_test]) #注入测试数据
        print ("测试数据：",example) #打印数据
        print ("测试标签：",label) #打印标签
    except KeyboardInterrupt:               #定义按ctrl+c键时，对应的异常处理
        print("程序终止...")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()