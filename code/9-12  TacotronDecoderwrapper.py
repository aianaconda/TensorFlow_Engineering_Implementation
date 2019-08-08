"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf

from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

attention = __import__("9-11  attention")
LocationSensitiveAttention = attention.LocationSensitiveAttention

class TacotronDecoderwrapper(tf.nn.rnn_cell.RNNCell):
  #初始化
  def __init__(self,encoder_outputs, is_training, rnn_cell, num_mels , outputs_per_step):

    super(TacotronDecoderwrapper, self).__init__()

    self._training = is_training
    self._attention_mechanism =  LocationSensitiveAttention(256, encoder_outputs)# [N, T_in, attention_depth=256]
    self._cell = rnn_cell
    self._frame_projection = tf.keras.layers.Dense(units=num_mels * outputs_per_step, name='projection_frame')# [N, T_out/r, M*r]

#    # [N, T_out/r, r]
    self._stop_projection = tf.keras.layers.Dense(units=outputs_per_step,name='projection_stop')
    self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

    self._output_size = num_mels * outputs_per_step#定义输出大小

  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size, self._attention_mechanism.batch_size,
      message=error_message)]

  @property
  def output_size(self):
      return self._output_size


  #@property
  def state_size(self):#返回的状态大小（代码参考AttentionWrapper）
    return tf.contrib.seq2seq.AttentionWrapperState(
      cell_state=self._cell._cell.state_size,
      time=tensor_shape.TensorShape([]),
      attention=self._attention_layer_size,
      alignments=self._attention_mechanism.alignments_size,
      alignment_history=(),#)#,
      attention_state = ())

  def zero_state(self, batch_size, dtype):#返回一个0状态（代码参考AttentionWrapper）
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
        "When calling zero_state of TacotronDecoderCell %s: " % self._base_name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the requested batch size.")
      with ops.control_dependencies(
        self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
          lambda s: array_ops.identity(s, name="checked_cell_state"),
          cell_state)

      return tf.contrib.seq2seq.AttentionWrapperState(
        cell_state=cell_state,
        time=array_ops.zeros([], dtype=tf.int32),
        attention=rnn_cell_impl._zero_state_tensors(self._attention_layer_size, batch_size, dtype),
        alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
        alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,dynamic_size=True),
        attention_state = tensor_array_ops.TensorArray(dtype=dtype, size=0,dynamic_size=True)
        )


  def __call__(self, inputs, state):#本时刻的真实输出y，decoder对上一时刻输出的状态。一起预测下一时刻

    drop_rate = 0.5 if self._training else 0.0#设置dropout
    #对输入预处理
    with tf.variable_scope('decoder_prenet'):# [N, T_in, prenet_depths[-1]=128]
        for i, size in enumerate([256, 128]):
            dense = tf.keras.layers.Dense(units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))(inputs)
            inputs = tf.keras.layers.Dropout( rate=drop_rate, name='dropout_%d' % (i+1))(dense, training=self._training)

    #加入注意力特征
    rnn_input = tf.concat([inputs, state.attention], axis=-1)

    #经过一个全连接变换。再传入解码器rnn中
    rnn_output, next_cell_state = self._cell(tf.keras.layers.Dense(256)(rnn_input), state.cell_state)

    #计算本次注意力
    context_vector, alignments, cumulated_alignments =attention_wrapper._compute_attention(self._attention_mechanism,
      rnn_output,state.alignments,None)#state.alignments为上一次的累计注意力

    #保存历史alignment(与原始的AttentionWrapper一致)
    alignment_history = state.alignment_history.write(state.time, alignments)

    #返回本次的wrapper状态
    next_state = tf.contrib.seq2seq.AttentionWrapperState( time=state.time + 1,
      cell_state=next_cell_state,attention=context_vector,
      alignments=cumulated_alignments, alignment_history=alignment_history,
      attention_state = state.attention_state)


    #计算本次结果：将解码器输出与注意力结果concat起来。作为最终的输入
    projections_input = tf.concat([rnn_output, context_vector], axis=-1)

    #两个全连接分别预测输出的下一个结果和停止标志<stop_token>
    cell_outputs = self._frame_projection(projections_input)#得到下一次outputs_per_step个帧的mel特征
    stop_tokens = self._stop_projection(projections_input)
    if self._training==False:
        stop_tokens = tf.nn.sigmoid(stop_tokens)

    return (cell_outputs, stop_tokens), next_state
