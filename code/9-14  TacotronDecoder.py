"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import collections
import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


#在输出类型中，添加token_output作为stop token的输出
class TacotronDecoderOutput(
		collections.namedtuple("TacotronDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
	pass

#自定义解码器实现类，来自于Tacotron 2的结构
class TacotronDecoder(decoder.Decoder):
    #初始化
	def __init__(self, cell, helper, initial_state, output_layer=None):
		rnn_cell_impl.assert_like_rnncell(type(cell), cell)
		if not isinstance(helper, helper_py.Helper):
			raise TypeError("helper must be a Helper, received: %s" % type(helper))
		if (output_layer is not None
				and not isinstance(output_layer, layers_base.Layer)):
			raise TypeError(
					"output_layer must be a Layer, received: %s" % type(output_layer))
		self._cell = cell
		self._helper = helper
		self._initial_state = initial_state
		self._output_layer = output_layer

	@property
	def batch_size(self):
		return self._helper.batch_size

	def _rnn_output_size(self):
		size = self._cell.output_size
		if self._output_layer is None:
			return size
		else:
			output_shape_with_unknown_batch = nest.map_structure(
					lambda s: tensor_shape.TensorShape([None]).concatenate(s),
					size)
			layer_output_shape = self._output_layer._compute_output_shape(
					output_shape_with_unknown_batch)
			return nest.map_structure(lambda s: s[1:], layer_output_shape)

	@property
	def output_size(self):
		return TacotronDecoderOutput(
				rnn_output=self._rnn_output_size(),
				token_output=self._helper.token_output_size,
				sample_id=self._helper.sample_ids_shape)

	@property
	def output_dtype(self):
		dtype = nest.flatten(self._initial_state)[0].dtype
		return TacotronDecoderOutput(
				nest.map_structure(lambda _: dtype, self._rnn_output_size()),
				tf.float32,
				self._helper.sample_ids_dtype)

	def initialize(self, name=None):
      #返回(finished, first_inputs, initial_state)
		return self._helper.initialize() + (self._initial_state,)

	def step(self, time, inputs, state, name=None):#执行解码的具体步骤

		with ops.name_scope(name, "TacotronDecoderStep", (time, inputs, state)):
			#调用解码器cell
			(cell_outputs, stop_token), cell_state = self._cell(inputs, state)

			#应用指定的输出层
			if self._output_layer is not None:
				cell_outputs = self._output_layer(cell_outputs)
			sample_ids = self._helper.sample(
					time=time, outputs=cell_outputs, state=cell_state)
            
            #调用help进行采样
			(finished, next_inputs, next_state) = self._helper.next_inputs(
					time=time,outputs=cell_outputs,state=cell_state,
					sample_ids=sample_ids,stop_token_preds=stop_token)

		outputs = TacotronDecoderOutput(cell_outputs, stop_token, sample_ids)
		return (outputs, next_state, next_inputs, finished)
