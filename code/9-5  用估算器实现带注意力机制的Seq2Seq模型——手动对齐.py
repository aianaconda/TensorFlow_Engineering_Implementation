# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from sklearn.model_selection import train_test_split

START_TOKEN = 0
END_TOKEN = 1

def seq2seq(mode, features, labels, params):
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    output_max_length = params['output_max_length']

    print("获得输入张量的名字",features.name,labels.name)
    inp = tf.identity(features[0], 'input_0')
    output = tf.identity(labels[0], 'output_0')
    print(inp.name,output.name)#用于钩子函数显示
    
    batch_size = tf.shape(features)[0]
    start_tokens = tf.tile([START_TOKEN], [batch_size])#也可以使用tf.zeros([batch_size], dtype=tf.int32)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), labels], 1)#为其添加开始标志
    
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(features, END_TOKEN)), 1,name="len")
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, END_TOKEN)), 1,name="outlen")
    
    input_embed = layers.embed_sequence( features, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    output_embed = layers.embed_sequence( train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)
    
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')

    Indcell = tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.IndRNNCell(num_units=num_units), "/device:GPU:0")
    IndyLSTM_cell = tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.IndyLSTMCell(num_units=num_units), "/device:GPU:0")
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([Indcell, IndyLSTM_cell])
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(multi_cell, input_embed,sequence_length=input_lengths, dtype=tf.float32)

    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)#decoder的长度会小，loss时不对齐
    
    
#    train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(output_embed, 
#                                                                       output_lengths),#output_lengths, 
#                                                                       embeddings, 0.3)
    
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens=tf.tile([START_TOKEN], [batch_size]), end_token=END_TOKEN)

    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
            
            #cell = tf.contrib.rnn.GRUCell(num_units=num_units)
            #cell = tf.contrib.IndRNN(num_units=num_units)
            cell = tf.contrib.rnn.IndRNNCell(num_units=num_units)
            if reuse == None:
                keep_prob=0.8
            else:
                keep_prob=1
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            
            
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=num_units / 2)
            
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size))
                #initial_state=encoder_final_state)
                
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=output_max_length
            )
            return outputs[0]
    train_outputs = decode(train_helper, 'decode')
    pred_outputs = decode(pred_helper, 'decode', reuse=True)

    
    
    print("train_outputs______",train_outputs,output_max_length)

    tf.identity(train_outputs.sample_id[0], name='train_pred')
    
    weights = tf.to_float(tf.not_equal(train_output[:, :-1], 0))#掩码
    masks = tf.sequence_mask(output_lengths,output_max_length,dtype=tf.float32,name="masks")

    train_outputs.rnn_output.get_shape()[1]
    
    pading = tf.ones([batch_size, output_max_length-tf.shape(train_outputs.rnn_output)[1],
                       tf.shape(train_outputs.rnn_output)[2]],dtype=tf.float32)
    print(tf.shape(pading))
    outforloss = tf.concat([train_outputs.rnn_output, pading], 1)
    print("_________",weights.get_shape(),outforloss.get_shape(),train_outputs.rnn_output.get_shape(),labels.get_shape())
#5    print("_________",weights.get_shape(),train_outputs.rnn_output.get_shape(),labels.get_shape())
    

    loss = tf.contrib.seq2seq.sequence_loss(
        outforloss, labels, weights=masks)
    
    train_op = layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer=params.get('optimizer', 'Adam'),
        learning_rate=params.get('learning_rate', 0.001),
        summaries=['loss', 'learning_rate'])

    tf.identity(pred_outputs.sample_id[0], name='predictions')
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_outputs.sample_id,
        loss=loss,
        train_op=train_op
    )




def get_formatter(keys, rev_vocab):

    def to_str(sequence):
        tokens = [
            rev_vocab.get(x, "<UNK>") for x in filter(lambda x:x!=END_TOKEN and x!= START_TOKEN,sequence)]
        return ' '.join(tokens)

    def format(values):
        res = []
        for key in keys:
            res.append("%s = %s" % (key, to_str(values[key])))
        return '\n'.join(res)
    return format




#def main():
import os
import jieba
path = "./chinese/"

alltext= []
for file in os.listdir(path):
    with open(path+file, 'r', encoding='UTF-8') as f:
        strtext = f.read().split('\n')  #按行读取，变为列表
        #strtext=list(filter(lambda x:x[0]!='-' and len(x)>0, strtext))
        strtext=list(filter( lambda x:len(x)>0, strtext))
#        print(strtext)        
        #strtext = list(map(lambda x:" ".join(jieba.cut(x))	,strtext[3:]))
        strtext = list(map(lambda x:" ".join(jieba.cut(x.replace('-','').replace(' ','')))	,strtext[3:]))
    
        print(file,strtext[:2])
#        break
        alltext = alltext+strtext
        print(len(alltext))

 
           
#过滤文本，选出5000个
top_k = 5000
#tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
#                                                  oov_token="<unk>", 
#                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                  oov_token="<unk>")
tokenizer.fit_on_texts(alltext)
#train_seqs = tokenizer.texts_to_sequences(alltext)
#print(train_seqs)
#
#构造字典
tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value <= top_k}
#print(tokenizer.word_index)
# putting <unk> token in the word2idx dictionary
tokenizer.word_index[tokenizer.oov_token] = top_k + 1
tokenizer.word_index['<start>'] = START_TOKEN
tokenizer.word_index['<end>'] = END_TOKEN
#print(tokenizer.word_index)

#反向字典
index_word = {value:key for key, value in tokenizer.word_index.items()}
print(len(index_word))

#变为向量
train_seqs = tokenizer.texts_to_sequences(alltext)
print(train_seqs[0])

inputseq,outseq = train_seqs[0::2], train_seqs[1::2];
print(len(inputseq))
print(len(outseq))

#按照最长的句子对齐。不足的在其后面补0
input_vector = tf.keras.preprocessing.sequence.pad_sequences(inputseq, padding='post',value=END_TOKEN)
output_vector = tf.keras.preprocessing.sequence.pad_sequences(outseq, padding='post',value=END_TOKEN)

print(len(input_vector))
start = np.zeros_like(input_vector[:,0])
start = np.reshape(start,[-1,1])

print(np.shape(start))
end = np.ones_like(input_vector[:,0])
end = np.reshape(end,[-1,1])
print(np.shape(start),np.shape(input_vector),np.shape(end))
input_vector = np.concatenate((input_vector,end),axis= 1)
output_vector = np.concatenate((output_vector,end),axis= 1)

print("in最大长度",len(input_vector[0]))
print("out最大长度",len(output_vector[0]))
in_max_length =len(input_vector[0]) 
out_max_length =len(output_vector[0])   

input_vector_train, input_vector_val, output_vector_train, output_vector_val = train_test_split(input_vector, 
                                                                    output_vector, 
                                                                    test_size=0.2, 
                                                                    random_state=0)

def to_str(sequence):
    tokens = [
        index_word.get(x, "<UNK>") for x in filter(lambda x:x!=END_TOKEN and x!=START_TOKEN ,sequence)]
    return ' '.join(tokens)

print(to_str(input_vector_train[0]))
print(to_str(output_vector_train[0]))

print(to_str(input_vector_val[0]))
print(to_str(output_vector_val[0]))
print(len(input_vector_val),len(output_vector_val))
print(input_vector_val[0])


BATCH_SIZE = 10
def train_input_fn(input_vector, output_vector, batch_size):  					#定义训练数据集输入函数
    #构造数据集的组成：一个特征输入，一个标签输入
    dataset = tf.data.Dataset.from_tensor_slices( (input_vector, output_vector) )  
    #dataset = tf.data.Dataset.from_tensor_slices( {'input_0':input_vector,'output_0': output_vector} ) 
    dataset = dataset.shuffle(1000).repeat().batch(batch_size, drop_remainder=True) 	#将数据集乱序、重复、批次划分. 
    return dataset  


params = {
        'vocab_size': len(index_word),
        'batch_size': BATCH_SIZE,
        'input_max_length': in_max_length,
        'output_max_length': out_max_length,
        'embed_dim': 100,
        'num_units': 256
    }
print(out_max_length,in_max_length)

model_dir='./modelrnn4'
est = tf.estimator.Estimator(  model_fn=seq2seq, model_dir=model_dir, params=params)



#打印过程信息
print_inputs = tf.train.LoggingTensorHook(
    ['input_0', 'output_0'], every_n_iter=100,
    formatter=get_formatter(['input_0', 'output_0'], index_word))
print_predictions = tf.train.LoggingTensorHook(
    ['predictions', 'train_pred'], every_n_iter=100,
    formatter=get_formatter(['predictions', 'train_pred'], index_word))

print_len = tf.train.LoggingTensorHook(
    ['len',"outlen","input_0","train_pred"], every_n_iter=100)


def feed_fn():
    index = np.random.randint(len(input_vector_val)-BATCH_SIZE)
    return {'IteratorGetNext:0':input_vector_val[index:index+BATCH_SIZE],
            'IteratorGetNext:1': output_vector_val[index:index+BATCH_SIZE]}



est.train(lambda: train_input_fn(input_vector_train, output_vector_train, BATCH_SIZE),
    hooks=[tf.train.FeedFnHook(feed_fn),
           print_inputs, print_predictions,print_len],steps=1000)


def wrapperFun(fn):									#定义装饰器函数
    def wrapper():										#包装函数
        return fn(input_vector_val, output_vector_val, BATCH_SIZE) 	#调用原函数
    return wrapper

@wrapperFun
def eval_input_fn(input_vector,labels, batch_size):  	#定义测试或应用模型时，数据集的输入函数
    #batch不允许为空
    assert batch_size is not None, "batch_size must not be None" 
      
    if labels is None:  							#如果评估，则没有标签
        inputs = input_vector  
    else:  
        inputs = (input_vector,labels)  

    #构造数据集 
    dataset = tf.data.Dataset.from_tensor_slices(inputs)  
    dataset = dataset.batch(batch_size, drop_remainder=True)  		#按批次划分
    return dataset     							#返回数据集   #返回数据集 
 
train_metrics = est.evaluate(input_fn=eval_input_fn)
print("train_metrics",train_metrics)
