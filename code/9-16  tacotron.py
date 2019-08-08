"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf

cbhg = __import__("9-10  cbhg")
encoder_cbhg = cbhg.encoder_cbhg
post_cbhg = cbhg.post_cbhg

rnnwrapper = __import__("9-12  TacotronDecoderwrapper")
TacotronDecoderwrapper = rnnwrapper.TacotronDecoderwrapper

Helpers= __import__("9-13  TacotronHelpers")
TacoTestHelper = Helpers.TacoTestHelper
TacoTrainingHelper = Helpers.TacoTrainingHelper

Decoder= __import__("9-14  TacotronDecoder")
TacotronDecoder = Decoder.TacotronDecoder

cn_dataset = __import__("9-15  cn_dataset")
symbols = cn_dataset.symbols



class Tacotron():
    #初始化
  def __init__(self, inputs,#形状为[N, input_length]，（n代表批次，input_length代表序列长度）
               input_lengths, #形状为[N]
               num_mels,outputs_per_step,num_freq,
               linear_targets=None,#形状为[N, targets_length, num_freq]，（targets_length代表输出序列）
               mel_targets=None, #形状为[N, targets_length, num_mels]，
               stop_token_targets=None):

    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]



      #词嵌入转换
      embedding_table = tf.get_variable( 'embedding', [len(symbols), 256], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)  #词嵌入形状为 [N, input_lengths, 256]

      #定义RNN编码网络（两层全连接+encoder_cbhg）
      drop_rate = 0.5 if is_training else 0.0
      with tf.variable_scope('encoder_prenet'):
        for i, size in enumerate([256, 128]):

           dense = tf.keras.layers.Dense(units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))(embedded_inputs)
           embedded_inputs = tf.keras.layers.Dropout(rate=drop_rate, name='dropout_%d' % (i+1))(dense,training=is_training)
      #最终解码特征输出的形状为[N, input_length, 256]
      encoder_outputs = encoder_cbhg(embedded_inputs, input_lengths, is_training, 256)

      # 定义RNN解码网络
      multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([
          tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.GRUCell(256)),
          tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.GRUCell(256))
        ], state_is_tuple=True)   #输出形状为 [N, input_length, 256]



      #实例化TacotronDecoderwrapper
      decoder_cell = TacotronDecoderwrapper(encoder_outputs,is_training, multi_rnn_cell,
                                            num_mels, outputs_per_step)

      if is_training:#选择不同的采样器
        helper = TacoTrainingHelper( mel_targets, num_mels, outputs_per_step)
      else:
        helper = TacoTestHelper(batch_size, num_mels, outputs_per_step)


      #初始化解码器状态
      decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      max_iters=300 #解码的最大长度为300，实际生成的长度为300*5
      (decoder_outputs, stop_token_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
         TacotronDecoder(decoder_cell, helper, decoder_init_state),maximum_iterations=max_iters)

      #对输出结果进行Reshape，生成mel特征[N, outputs_per_step, num_mels]。
      self.mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, num_mels])
      self.stop_token_outputs = tf.reshape(stop_token_outputs, [batch_size, -1])

      #用 CBHG对mel特征后处理，形状为[N, outputs_per_step, 256]
      post_outputs = post_cbhg(self.mel_outputs, num_mels, is_training, 256)
      #用全连接网络将处理后的mel特征还原，输出形状为[N, outputs_per_step, num_freq]
      self.linear_outputs = tf.keras.layers.Dense( num_freq)(post_outputs)

      #获取注意力的全部结果，用于可视化
      self.alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      self.stop_token_targets = stop_token_targets
      tf.logging.info('Initialized Tacotron model. Dimensions: ')
      tf.logging.info('  embedding:               {}'.format(embedded_inputs.shape))
      tf.logging.info('  encoder out:             {}'.format(encoder_outputs.shape))
      tf.logging.info('  decoder out (r frames):  {}'.format(decoder_outputs.shape))
      tf.logging.info('  decoder out (1 frame):   {}'.format(self.mel_outputs.shape))
      tf.logging.info('  postnet out:             {}'.format(post_outputs.shape))
      tf.logging.info('  linear out:              {}'.format(self.linear_outputs.shape))
      tf.logging.info('  stop token:              {}'.format(self.stop_token_outputs.shape))


  def buildTrainModel(self,sample_rate,num_freq,global_step):
    #计算loss
    with tf.variable_scope('loss') as scope:
        #计算mel特征 loss
      self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      #计算停止符loss
      self.stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                            labels=self.stop_token_targets,
                                            logits=self.stop_token_outputs))

      l1 = tf.abs(self.linear_targets - self.linear_outputs)
      #优先考虑4000 Hz以下频率的损失.
      n_priority_freq = int(4000 / (sample_rate * 0.5) * num_freq)
      self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])

      self.loss = self.mel_loss + self.linear_loss + self.stop_token_loss

      #定义优化器
    with tf.variable_scope('optimizer') as scope:
      initial_learning_rate=0.001
      self.learning_rate = _learning_rate_decay(initial_learning_rate, global_step)

      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      #在BN运算之后，更新权重
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)

#退化学习率
def _learning_rate_decay(init_lr, global_step):
  warmup_steps = 4000.0#超参方法来自于tensor2tensor:
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
