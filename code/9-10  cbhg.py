"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf



def highwaynet(inputs, scope, depth): #定义高速通道函数
  with tf.variable_scope(scope):
    H = tf.keras.layers.Dense(units=depth,activation='relu',name='H')(inputs)
    T = tf.keras.layers.Dense(units=depth,activation='sigmoid',name='T',
                        bias_initializer=tf.constant_initializer(-1.0))(inputs)
    return H * T + inputs * (1.0 - T)

def cbhg(inputs, input_lengths, is_training, scope, K, projections, depth):
  with tf.variable_scope(scope):
    with tf.variable_scope('conv_bank'):
      conv_bank = []
      for k in range(1,K+1):#使用了same卷积。输出结果的尺度与卷积核无关，只与步长有关
          con1d_output = tf.keras.layers.Conv1D(128,k,activation=tf.nn.relu,
                                    padding='same',name = 'conv1d_%d'% k)(inputs)

          con1d_output_bn = tf.keras.layers.BatchNormalization(name = 'conv1d_%d_bn'% k)(con1d_output,
                                                          training=is_training)
          conv_bank.append(con1d_output_bn)
      conv_outputs = tf.concat(conv_bank,axis=-1)

    #最大池化层:
    maxpool_output = tf.keras.layers.MaxPool1D(pool_size=2,strides=1,padding='same')(conv_outputs)
    #使用2层卷积进行维度变化
    proj1_output = tf.keras.layers.Conv1D(projections[0],3,activation=tf.nn.relu,
                                    padding='same',name = 'proj_1')(maxpool_output)
    proj1_output_bn = tf.keras.layers.BatchNormalization(name = 'proj_1_bn')(proj1_output, training=is_training)

    proj2_output = tf.keras.layers.Conv1D(projections[1],3, padding='same',name = 'proj_2')(proj1_output_bn)
    proj2_output_bn = tf.keras.layers.BatchNormalization(name = 'proj_2_bn')(proj2_output, training=is_training)





    #残差连接:
    highway_input = proj2_output_bn + inputs

    half_depth = depth // 2 #必须被2整除，之后的结果是每个方向RNN的cell个数
    assert half_depth*2 == depth, 'depth必须被2整除.'

    #调整残差后的维度，与RNN的cell个数一致:
    if highway_input.shape[2] != half_depth:
      highway_input = tf.keras.layers.Dense( half_depth)(highway_input)

    # 4层高速通道:
    for i in range(4):
      highway_input = highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
    rnn_input = highway_input

    #双向RNN

    outputs, states = tf.nn.bidirectional_dynamic_rnn(tf.keras.layers.GRUCell(half_depth),tf.keras.layers.GRUCell(half_depth),
                      rnn_input, sequence_length=input_lengths,dtype=tf.float32)


    return tf.concat(outputs, axis=2)  #将双向RNN正反方向结果组合到一起

#用于编码器中的CBHG
def encoder_cbhg(inputs, input_lengths, is_training, depth):#depth为RNN单元个数，由于是双向，必须被二整除
  return cbhg(inputs,input_lengths, is_training,scope='encoder_cbhg', K=16,
  projections=[ 128, inputs.shape.as_list()[2] ], depth=depth)

#用于解码器中的CBHG
def post_cbhg(inputs, input_dim, is_training, depth):
  return cbhg( inputs,None, is_training, scope='post_cbhg', K=8,projections=[256, input_dim], depth=depth)




