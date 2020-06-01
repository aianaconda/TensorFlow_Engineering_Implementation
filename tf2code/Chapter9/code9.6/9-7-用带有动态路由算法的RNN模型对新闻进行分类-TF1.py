# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import tensorflow as tf
import numpy as np


#定义参数
num_words = 20000
maxlen = 80

#加载数据  自路透社的11,228条新闻，分为了46个主题
print('Loading data...')
(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.reuters.load_data(path='./reuters.npz',num_words=num_words)
#print(len(x_train), 'train sequences')
#print(len(x_test), 'test sequences')
#print(x_train[0])
#print(y_train[:10])

#word_index = tf.keras.datasets.reuters.get_word_index('./reuters_word_index.json')# 单词--下标 对应字典
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])# 下标-单词对应字典
#
#decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]]) 
#print(decoded_newswire)



#数据对齐
x_train =  tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,padding = 'post')
x_test =  tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,padding = 'post' )
print('Pad sequences x_train shape:', x_train.shape)

leng = np.count_nonzero(x_train,axis = 1)#获取长度
print(leng[:3])


tf.reset_default_graph()


BATCH_SIZE = 100#批次
#定义数据集
dataset = tf.data.Dataset.from_tensor_slices(((x_train,leng), y_train)).shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)



def mkMask(input_tensor, maxLen): #计算变长RNN的掩码
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)

#定义函数，将输入转化成uhat
def shared_routing_uhat(caps,          #输入 shape(b_sz, maxlen, caps_dim)
                        out_caps_num,  #输出胶囊个数
                        out_caps_dim, scope=None): #输出胶囊维度
                        
    batch_size,maxlen = tf.shape(caps)[0],tf.shape(caps)[1] #获取批次和长度

    with tf.variable_scope(scope or 'shared_routing_uhat'):#转成uhat
        caps_uhat = tf.layers.dense(caps, out_caps_num * out_caps_dim, activation=tf.tanh)
        caps_uhat = tf.reshape(caps_uhat, shape=[batch_size, maxlen, out_caps_num, out_caps_dim])

    return caps_uhat #输出batch_size, maxlen, out_caps_num, out_caps_dim


def masked_routing_iter(caps_uhat, seqLen, iter_num): #动态路由计算

    assert iter_num > 0
    batch_size,maxlen = tf.shape(caps_uhat)[0],tf.shape(caps_uhat)[1] #获取批次和长度
    out_caps_num = int(caps_uhat.get_shape()[2])
    seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)
    mask = mkMask(seqLen, maxlen)     # shape(batch_size, maxlen)
    floatmask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)    # shape(batch_size, maxlen, 1)

    # shape(b_sz, maxlen, out_caps_num)
    B = tf.zeros([batch_size, maxlen, out_caps_num], dtype=tf.float32)
    for i in range(iter_num):
        C = tf.nn.softmax(B, axis=2)  # shape(batch_size, maxlen, out_caps_num)
        C = tf.expand_dims(C*floatmask, axis=-1)  # shape(batch_size, maxlen, out_caps_num, 1)
        weighted_uhat = C * caps_uhat   # shape(batch_size, maxlen, out_caps_num, out_caps_dim)

        S = tf.reduce_sum(weighted_uhat, axis=1)    # shape(batch_size, out_caps_num, out_caps_dim)

        V = _squash(S, axes=[2])  # shape(batch_size, out_caps_num, out_caps_dim)
        V = tf.expand_dims(V, axis=1)   # shape(batch_size, 1, out_caps_num, out_caps_dim)
        B = tf.reduce_sum(caps_uhat * V, axis=-1) + B   # shape(batch_size, maxlen, out_caps_num)

    V_ret = tf.squeeze(V, axis=[1])  # shape(batch_size, out_caps_num, out_caps_dim)
    S_ret = S
    return V_ret, S_ret



def _squash(in_caps, axes):#定义_squash激活函数
    _EPSILON = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(in_caps), axis=axes, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + _EPSILON)
    vec_squashed = scalar_factor * in_caps  # element-wise
    return vec_squashed

#定义函数，使用动态路由对RNN结果信息聚合
def routing_masked(in_x, xLen, out_caps_dim, out_caps_num, iter_num=3,
                                dropout=None, is_train=False, scope=None):
    assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
    b_sz = tf.shape(in_x)[0]
    with tf.variable_scope(scope or 'routing'):
        caps_uhat = shared_routing_uhat(in_x, out_caps_num, out_caps_dim, scope='rnn_caps_uhat')
        attn_ctx, S = masked_routing_iter(caps_uhat, xLen, iter_num)
        attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num*out_caps_dim])
        if dropout is not None:
            attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
    return attn_ctx












x = tf.placeholder("float", [None, maxlen]) #定义输入占位符
x_len = tf.placeholder(tf.int32, [None, ])#定义输入序列长度占位符
y = tf.placeholder(tf.int32, [None, ])#定义输入分类标签占位符

nb_features = 128   #词嵌入维度  
embeddings = tf.keras.layers.Embedding(num_words, nb_features)(x)

#定义带有IndyLSTMCell的RNN网络
hidden = [100,50,30]#RNN单元个数
stacked_rnn = []
for i in range(3):
    cell = tf.contrib.rnn.IndyLSTMCell(hidden[i])
    stacked_rnn.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8))
mcell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn)

rnnoutputs,_  = tf.nn.dynamic_rnn(mcell,embeddings,dtype=tf.float32)
out_caps_num = 5 #定义输出的胶囊个数
n_classes = 46#分类个数

outputs = routing_masked(rnnoutputs, x_len,int(rnnoutputs.get_shape()[-1]), out_caps_num, iter_num=3)
print(outputs.get_shape())
pred =tf.layers.dense(outputs,n_classes,activation = tf.nn.relu)



#定义优化器
learning_rate = 0.001
cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

iterator1 = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
one_element1 = iterator1.get_next()							#获取一个元素


#训练网络
with tf.Session()  as sess:
    sess.run( iterator1.make_initializer(dataset) )		#初始化迭代器
    sess.run(tf.global_variables_initializer())
    EPOCHS = 20
    for ii in range(EPOCHS):
        alloss = []  									#数据集迭代两次
        while True:											#通过for循环打印所有的数据
            try:
                inp, target = sess.run(one_element1)
                _,loss =sess.run([optimizer,cost], feed_dict={x: inp[0],x_len:inp[1], y: target})
                alloss.append(loss)

            except tf.errors.OutOfRangeError:
                #print("遍历结束")
                print("step",ii+1,": loss=",np.mean(alloss))
                sess.run( iterator1.make_initializer(dataset) )	#从头再来一遍
                break




#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
##    step = 1
#    EPOCHS = 2000000
#    print("ok")
#    for epoch in range(EPOCHS):
#        alloss = []
#        inp, target = sess.run(one_element)
##        if len(inp[0])!= BATCH_SIZE:
##            print(len(inp))
##            continue
#        _,loss =sess.run([optimizer,cost], feed_dict={x: inp[0],x_len:inp[1], y: target})
#        alloss.append(loss)
#        #print(loss)
#        if epoch%100==0:
#            print(np.mean(alloss))

#
#
#x = tf.placeholder("float", [None, sequence_length, nb_features])
#y = tf.placeholder(tf.int32, [None, nb_out])x = tf.placeholder("float", [None, sequence_length, nb_features])
#y = tf.placeholder(tf.int32, [None, nb_out])
#
#
##生成词向量
#embeddings = tf.keras.layers.Embedding(num_words, 128)(S_inputs)
##vectorized sequences
#def vectorize_sequences(sequences,dimension=10000):
#    results = np.zeros((len(sequences),dimension))
#    for i ,sequence in enumerate(sequences):
#        results[i,sequence] = 1
#    return results
# 
##preparing the data
##encoding the data
#x_train = vectorize_sequences(train_data)
#x_test = vectorize_sequences(test_data)
# 
## #one-hot encoding the labels
## def to_one_hot(labels,dimension=46):
##     results = np.zeros((len(labels),dimension))
##     for i,label in enumerate(labels):
##         results[i, label] = 1.
##     return results
## one_hot_train_labels = to_one_hot(train_labels)
## one_hot_test_labels = to_one_hot(test_labels)
# 
##using keras build-in methos to change to one-hot labels
#one_hot_train_labels = to_categorical(train_labels)
#one_hot_test_labels = to_categorical(test_labels)
# 
##model setup
#model = models.Sequential()
#model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
#model.add(layers.Dense(64,activation='relu'))
#model.add(layers.Dense(46,activation='softmax'))
# 
##model compile
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# 
##validating our apporoach
#x_val = x_train[:1000]
#partial_x_train = x_train[1000:]
#y_val = one_hot_train_labels[:1000]
#partial_y_train = one_hot_train_labels[1000:]
# 
##training the model
#history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# 
##ploting the training and validation loss
#loss = history.history['loss']
#val_loss  = history.history['val_loss']
#epochs = range(1,len(loss)+1)
#plt.plot(epochs,loss,'bo',label='Training loss')
#plt.plot(epochs,val_loss,'b',label='Validating loss')
#plt.title('Training and Validating loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()
# 
##ploting the training and validation accuracy
#plt.clf()
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#plt.plot(epochs,acc,'ro',label='Training acc')
#plt.plot(epochs,val_acc,'r',label='Validating acc')
#plt.title('Training and Validating accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('accuracy')
#plt.legend()
#plt.show()
# 
##evaluate
#final_result = model.evaluate(x_test,one_hot_test_labels)
#print(final_result)
