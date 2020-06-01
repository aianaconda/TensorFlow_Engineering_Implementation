"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
import tensorflow.keras.layers as kl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
tf.compat.v1.disable_v2_behavior()



#读入PM_train数据样本
train_df = pd.read_csv('./PM_train.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id','cycle'])

#读入PM_test数据样本
test_df = pd.read_csv('./PM_test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

#读入PM_truth数据样本
truth_df = pd.read_csv('./PM_truth.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)


#处理训练数据
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)


w0 = 15#定义了两个分类参数，15周期与30周期
w1 = 30 

train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2


train_df['cycle_norm'] = train_df['cycle']#训练数据归一化
train_df['RUL_norm'] = train_df['RUL']
cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)

#合成训练数据特征列
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)


#处理测试数据
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)

#生成测试数据的RUL
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

#生成测试标签
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2


test_df['cycle_norm'] = test_df['cycle']#对测试数据归一化
test_df['RUL_norm'] = test_df['RUL']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)

sequence_length = 50  #定义序列长度

def gen_sequence(id_df, seq_length, seq_cols):#按照sequence_length获得序列数据
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
        
#合成特征列
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)


seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
           for id in train_df['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)#生成训练数据
print(seq_array.shape)


def gen_labels(id_df, seq_length, label):#生成标签
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

#生成训练分类标签
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1']) 
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape

#生成训练回归标签
labelreg_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['RUL_norm']) 
             for id in train_df['id'].unique()]

labelreg_array = np.concatenate(labelreg_gen).astype(np.float32)
print(labelreg_array.shape)

#从测试数据中，找到序列长度大于sequence_length的数据，将其最后sequence_length个数据取出
seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)#生成测试数据

y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
#生成分类回归标签
label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
#生成测试回归标签
labelreg_array_test_last = test_df.groupby('id')['RUL_norm'].nth(-1)[y_mask].values
labelreg_array_test_last = labelreg_array_test_last.reshape(labelreg_array_test_last.shape[0],1).astype(np.float32)





BATCH_SIZE = 80#指定批次
#定义训练数据集
dataset = tf.data.Dataset.from_tensor_slices((seq_array, (label_array,labelreg_array))).shuffle(1000)
dataset = dataset.repeat().batch(BATCH_SIZE)



#测试数据集
testdataset = tf.data.Dataset.from_tensor_slices((seq_array_test_last, (label_array_test_last,labelreg_array_test_last)))
testdataset = testdataset.batch(BATCH_SIZE, drop_remainder=True)


import JANetLSTMCell
tf.compat.v1.reset_default_graph()
learning_rate = 0.001
units = 100#GRU单元个数
#构建网络节点
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]
reg_out= labelreg_array.shape[1]
n_classes = 2
x = tf.compat.v1.placeholder("float", [None, sequence_length, nb_features])
y = tf.compat.v1.placeholder(tf.int32, [None, nb_out])
yreg = tf.compat.v1.placeholder("float", [None, reg_out])
#
hidden = [100,50,36]

cell1=JANetLSTMCell.JANetLSTMCell(hidden[0], t_max=sequence_length,recurrent_dropout=0.8)
rnn=kl.RNN(cell=cell1,return_sequences=True)(x)
cell2=JANetLSTMCell.JANetLSTMCell(hidden[1], recurrent_dropout=0.8)
rnn=kl.RNN(cell=cell2,return_sequences=True)(rnn)
cell3=JANetLSTMCell.JANetLSTMCell(hidden[2], recurrent_dropout=0.8)
rnn=kl.RNN(cell=cell3,return_sequences=True)(rnn)

outputs = rnn
outputs = tf.transpose(a=outputs, perm=[1, 0, 2])
print(outputs.get_shape())

pred =kl.Conv2D(n_classes,6,activation = 'relu')(tf.reshape(outputs[-1],[-1,6,6,1]))
pred =tf.reshape(pred,(-1,n_classes))#分类模型

predreg =kl.Conv2D(1,1,activation = 'sigmoid')(tf.reshape(outputs[-1],[-1,1,1,36]))
predreg =tf.reshape(predreg,(-1,1))#回归模型

costreg = tf.reduce_mean(input_tensor=abs(predreg - yreg))
costclass = tf.reduce_mean(input_tensor=tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=pred, labels=y))

cost =(costreg+costclass)/2
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)			#生成一个训练集的迭代器
one_element = iterator.get_next()

iterator_test = tf.compat.v1.data.make_one_shot_iterator(testdataset)			#生成一个测试集的迭代器
one_element_test = iterator_test.get_next()	

EPOCHS = 5000    #指定迭代次数
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(EPOCHS):#训练模型
        alloss = []
        inp, (target,targetreg) = sess.run(one_element)
        if len(inp)!= BATCH_SIZE:
            continue
        predregv,_,loss =sess.run([predreg,optimizer,cost], feed_dict={x: inp, y: target,yreg:targetreg})
        
        alloss.append(loss)
        if epoch%100==0:
            print(np.mean(alloss))

#测试模型
    alloss = []	#收集loss值
    while True:	
        try:
            inp, (target,targetreg) = sess.run(one_element_test)
            predv,predregv,loss =sess.run([pred,predreg,cost], feed_dict={x: inp, y: target,yreg:targetreg})
            alloss.append(loss)
            print("分类结果：",target[:20,0],np.argmax(predv[:20],axis = 1))
            print("回归结果：",np.asarray(targetreg[:20]*train_df['RUL'].max()+train_df['RUL'].min(),np.int32)[:,0],
                  np.asarray(predregv[:20]*train_df['RUL'].max()+train_df['RUL'].min(),np.int32)[:,0])
            print(loss)
            
        except tf.errors.OutOfRangeError:
            print("测试结束")
            #可视化显示
            y_true_test =np.asarray(targetreg*train_df['RUL'].max()+train_df['RUL'].min(),np.int32)[:,0]         
            y_pred_test = np.asarray(predregv*train_df['RUL'].max()+train_df['RUL'].min(),np.int32)[:,0]

            fig_verify = plt.figure(figsize=(12, 8))
            plt.plot(y_pred_test, color="blue")
            plt.plot(y_true_test, color="green")
            plt.title('prediction')
            plt.ylabel('value')
            plt.xlabel('row')
            plt.legend(['predicted', 'actual data'], loc='upper left')
            plt.show()
            fig_verify.savefig("./model_regression_verify.png") 
            print(np.mean(alloss))
            break
