# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import sys
nets_path = r'slim'                             #加载环境变量
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')

import tensorflow as tf
from nets.nasnet import nasnet                 #导出nasnet
slim = tf.contrib.slim

import os
mydataset = __import__("5-1  mydataset")
creat_dataset_fromdir = mydataset.creat_dataset_fromdir

class MyNASNetModel(object):
    """微调模型类MyNASNetModel
    """
    def __init__(self, model_path=''):
        self.model_path = model_path  #原始模型的路径

    def MyNASNet(self,images,is_training):
        arg_scope = nasnet.nasnet_mobile_arg_scope()          #获得模型命名空间
        with slim.arg_scope(arg_scope):
            #构建NASNet Mobile模型
            logits, end_points = nasnet.build_nasnet_mobile(images,num_classes = self.num_classes+1,
                                                            is_training=is_training)

        global_step = tf.train.get_or_create_global_step()  #定义记录步数的张量

        return logits,end_points,global_step   #返回有用的张量

    def FineTuneNASNet(self,is_training):       #实现微调模型的网络操作
        model_path = self.model_path

        exclude = ['final_layer','aux_7']  #恢复超参， 除了exclude以外的全部恢复
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        if is_training == True:
            init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore,ignore_missing_vars=True)
        else:
            init_fn = None

        tuning_variables = [] #将没有恢复的超参收集起来，用于微调训练
        for v in exclude:
            tuning_variables += slim.get_variables(v)

        print('final_layer:',slim.get_variables('final_layer'))
        print('aux_7:',slim.get_variables('aux_7'))
        print("tuning_variables:",tuning_variables)

        return init_fn,tuning_variables



    def build_acc_base(self,labels):#定义评估函数
        #返回张量中最大值的索引
        self.prediction = tf.cast(tf.argmax(self.logits, 1),tf.int32)
        #计算prediction、labels是否相同
        self.correct_prediction = tf.equal(self.prediction, labels)
        #计算平均值
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
        #计算这些目标是否在最高的前5预测中，并取平均值
        self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=labels, k=5),tf.float32))

    def load_cpk(self,global_step,sess,begin = 0,saver= None,save_path = None):
        """
        模型储存和导出
        """
        if begin == 0:
            save_path=r'./train_nasnet'  #定义检查点路径
            if not os.path.exists(save_path):
                print("there is not a model path:",save_path)
            saver = tf.train.Saver(max_to_keep=1) # 生成saver
            return saver,save_path
        else:
            kpt = tf.train.latest_checkpoint(save_path)#查找最新的检查点
            print("load model:",kpt)
            startepo= 0#计步
            if kpt!=None:
                saver.restore(sess, kpt) #还原模型
                ind = kpt.find("-")
                startepo = int(kpt[ind+1:])
                print("global_step=",global_step.eval(),startepo)
            return startepo

    def build_model_train(self,images, labels,learning_rate1,learning_rate2,is_training):
        self.logits,self.end_points, self.global_step= self.MyNASNet(images,is_training=is_training)
        self.step_init = self.global_step.initializer
        self.init_fn,self.tuning_variables = self.FineTuneNASNet(is_training=is_training)
        #定义损失函数
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)
        loss = tf.losses.get_total_loss()

        #定义退化学习速率
        learning_rate1 = tf.train.exponential_decay(learning_rate=learning_rate1,#微调的学习率
                global_step=self.global_step,
                decay_steps=100, decay_rate=0.5)
        learning_rate2 = tf.train.exponential_decay(learning_rate=learning_rate2,#联调的学习率
                global_step=self.global_step,
                decay_steps=100, decay_rate=0.2)

        #定义冲量Momentum优化器
#        last_optimizer = tf.train.MomentumOptimizer(learning_rate1, 0.8, use_nesterov=True)
#        full_optimizer = tf.train.MomentumOptimizer(learning_rate2, 0.8, use_nesterov=True)

        last_optimizer = tf.train.AdamOptimizer(learning_rate1)
        full_optimizer = tf.train.AdamOptimizer(learning_rate2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):#更新批量归一化中的参数
            #使loss减小方向做优化
            self.last_train_op = last_optimizer.minimize(loss, self.global_step,var_list=self.tuning_variables)
            self.full_train_op = full_optimizer.minimize(loss, self.global_step)

        #self.opt_init = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
#        self.opt_init = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]


        self.build_acc_base(labels)#定义评估模型相关指标

        tf.summary.scalar('accuracy', self.accuracy)#写入日志，支持tensorBoard操作
        tf.summary.scalar('accuracy_top_5', self.accuracy_top_5)

        #将收集的所有默认图表并合并
        self.merged = tf.summary.merge_all()
        #写入日志文件
        self.train_writer = tf.summary.FileWriter('./log_dir/train')
        self.eval_writer = tf.summary.FileWriter('./log_dir/eval')

        self.saver,self.save_path = self.load_cpk(self.global_step,None)   #定义检查点相关变量



    def build_model(self,mode='train',testdata_dir='./data/val',traindata_dir='./data/train', batch_size=32,learning_rate1=0.001,learning_rate2=0.001):

        if mode == 'train':
            tf.reset_default_graph()
            #创建训练数据和测试数据的Dataset数据集
            dataset,self.num_classes = creat_dataset_fromdir(traindata_dir,batch_size)
            testdataset,_ = creat_dataset_fromdir(testdata_dir,batch_size,isTrain = False)

            #创建一个可初始化的迭代器
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            #读取数据
            images, labels = iterator.get_next()
            iterator.make_initializer

            self.train_init_op = iterator.make_initializer(dataset)
            self.test_init_op = iterator.make_initializer(testdataset)

            self.build_model_train(images, labels,learning_rate1,learning_rate2,is_training=True)

            self.global_init = tf.global_variables_initializer()
            tf.get_default_graph().finalize()


        elif mode == 'test':
            tf.reset_default_graph()

            #创建测试数据的Dataset数据集
            testdataset,self.num_classes = creat_dataset_fromdir(testdata_dir,batch_size,isTrain = False)

            #创建一个可初始化的迭代器
            iterator = tf.data.Iterator.from_structure(testdataset.output_types, testdataset.output_shapes)
            #读取数据
            self.images, labels = iterator.get_next()

            self.test_init_op = iterator.make_initializer(testdataset)
            self.logits,self.end_points, self.global_step= self.MyNASNet(self.images,is_training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)   #定义检查点相关变量
            #评估指标
            self.build_acc_base(labels)
            tf.get_default_graph().finalize()


        elif mode == 'eval':
            tf.reset_default_graph()
            #创建测试数据的Dataset数据集
            testdataset,self.num_classes = creat_dataset_fromdir(testdata_dir,batch_size,isTrain = False)

            #创建一个可初始化的迭代器
            iterator = tf.data.Iterator.from_structure(testdataset.output_types, testdataset.output_shapes)
            #读取数据
            self.images, labels = iterator.get_next()


            self.logits,self.end_points, self.global_step= self.MyNASNet(self.images,is_training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)   #定义检查点相关变量
            tf.get_default_graph().finalize()
