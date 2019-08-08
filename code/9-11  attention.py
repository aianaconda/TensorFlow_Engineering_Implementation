"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
#from tensorflow.python.layers import core as layers_core
#from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope
from tensorflow.python.ops import array_ops, variable_scope

def _location_sensitive_score(processed_query, processed_location, keys):
	#获取注意力的深度（全连接神经元的个数）
	dtype = processed_query.dtype
	num_units = keys.shape[-1].value or array_ops.shape(keys)[-1]

   #定义了最后一个全连接v
	v_a = tf.get_variable('attention_variable', shape=[num_units], dtype=dtype,
		initializer=tf.contrib.layers.xavier_initializer())
    
   #定义了偏执b
	b_a = tf.get_variable('attention_bias', shape=[num_units], dtype=dtype,
		initializer=tf.zeros_initializer())
    #计算注意力分数
	return tf.reduce_sum(v_a * tf.tanh(keys + processed_query + processed_location + b_a), [2])

def _smoothing_normalization(e):#平滑归一化函数,返回[batch_size, max_time]，代替softmax
	return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class LocationSensitiveAttention(BahdanauAttention):#位置敏感注意力
	
	def __init__(self,                 #初始化
			num_units,                   #实现过程中全连接的神经元个数
			memory,                     #编码器encoder的结果
			smoothing=False,             #是否使用平滑归一化函数代替softmax
			cumulate_weights=True,       #是否对注意力结果进行累加
			name='LocationSensitiveAttention'):
		

		#smoothing为true则使用_smoothing_normalization，否则使用softmax
		normalization_function = _smoothing_normalization if (smoothing == True) else None
		super(LocationSensitiveAttention, self).__init__(
				num_units=num_units,
				memory=memory,
				memory_sequence_length=None,
				probability_fn=normalization_function,#当为None时，基类会调用softmax
				name=name)

		self.location_convolution = tf.layers.Conv1D(filters=32,
			kernel_size=(31, ), padding='same', use_bias=True,
			bias_initializer=tf.zeros_initializer(), name='location_features_convolution')
		self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
			dtype=tf.float32, name='location_features_layer')
		self._cumulate = cumulate_weights

	def __call__(self, query, #query为解码器decoder的中间态结果[batch_size, query_depth]
                      state):#state为上一次的注意力[batch_size, alignments_size]
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			#全连接处理query特征[batch_size, query_depth] -> [batch_size, attention_dim]
			processed_query = self.query_layer(query) if self.query_layer else query
			#维度扩展  -> [batch_size, 1, attention_dim]
			processed_query = tf.expand_dims(processed_query, 1)

			#维度扩展 [batch_size, max_time] -> [batch_size, max_time, 1]
			expanded_alignments = tf.expand_dims(state, axis=2)
			#通过卷积获取位置特征[batch_size, max_time, filters]
			f = self.location_convolution(expanded_alignments)
			#经过全连接变化[batch_size, max_time, attention_dim]
			processed_location_features = self.location_layer(f)

			#计算注意力分数 [batch_size, max_time]
			energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)


		#计算最终注意力结果[batch_size, max_time]
		alignments = self._probability_fn(energy, state)

		#是否累加
		if self._cumulate:
			next_state = alignments + state
		else:
			next_state = alignments#[batch_size, alignments_size],alignments_size为memory的最大序列次数max_time

		return alignments, next_state
