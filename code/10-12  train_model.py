"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
train_a_sequence = __import__("10-11  train_a_sequence")

tf.logging.set_verbosity(tf.logging.INFO)   

#定义相关参数
DATA_DIR ='./no'  #定义载入的样本路径

TRAIN_FROM_CKPT =False   #是否从检查点开始训练

DYNAMIC_BATCH = False  #是否使用动态批次
BATCH_SIZE = 256       #定义批次大小

SCHEDULE_ITERATIONS = True  #是否根据长度调整训练次数
SCHEDULE_MULT = 200         #每个长度增加的训练次数
ITERATIONS_PER_SEQ_LENGTH = 2000  #定义每个长度训练时的迭代次数

REAL_BATCH_SIZE = BATCH_SIZE

START_SEQ = 1    #待训练的起始长度
END_SEQ  = 256  #最终长度

#开始训练
stages = range(START_SEQ, END_SEQ)  
printstr = '------------------Stages : ' + ' '.join(map(str, stages)) + "--------------"
tf.logging.info(printstr)

for i in range(len(stages)): #从START_SEQ开始依次对每个长度的模型进行训练
    prev_seq_length = stages[i-1] if i>0 else 0  #定义变量，用于获得上次模型的路径名称
    seq_length = stages[i]
    
    printstr = "------------------Training on Seq Len = %d, BATCH SIZE: %d------------------" % (seq_length,
                                                                                                 BATCH_SIZE)
    tf.logging.info(printstr)
    
    tf.reset_default_graph()
    
    if SCHEDULE_ITERATIONS:  #计算本次长度训练的迭代次数
        iterations = min((seq_length + 1) * SCHEDULE_MULT, ITERATIONS_PER_SEQ_LENGTH)
    else:
        iterations = ITERATIONS_PER_SEQ_LENGTH

    is_first = seq_length == stages[0] and not (TRAIN_FROM_CKPT)
    #开始训练
    train_a_sequence.run( iterations, seq_length,is_first,BATCH_SIZE , prev_seq_length,DATA_DIR,END_SEQ )

    if DYNAMIC_BATCH:
        BATCH_SIZE = REAL_BATCH_SIZE / seq_length
