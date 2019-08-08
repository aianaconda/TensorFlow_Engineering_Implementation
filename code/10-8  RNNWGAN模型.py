"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

#定义网络参数
DISC_STATE_SIZE = 256#定义判别器中RNNcell节点个数
GEN_STATE_SIZE = 256#定义生成器中RNNcell节点个数
GEN_RNN_LAYERS = 1
LAMBDA = 10.0  #惩罚参数

#定义判别器函数
def Discriminator_RNN (inputs, charmap_len, seq_len, reuse=False, rnn_cell=None):
    with tf.variable_scope("Discriminator", reuse=reuse):
        flat_inputs = tf.reshape(inputs, [-1, charmap_len])
        
        weight = tf.get_variable("embedding", shape=[charmap_len, DISC_STATE_SIZE],
            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        
        #通过全链接转成与RNN同样维度的向量
        inputs = tf.reshape(flat_inputs@weight, [-1, seq_len, DISC_STATE_SIZE])
        inputs = tf.unstack(tf.transpose(inputs, [1,0,2]))
        #输入RNN网络
        cell = rnn_cell(DISC_STATE_SIZE)
        output, state = tf.contrib.rnn.static_rnn(cell,inputs,dtype=tf.float32)

        weight = tf.get_variable("W", shape=[DISC_STATE_SIZE, 1],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        bias = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        #通过全链接网络生成判别结果
        prediction = output[-1]@weight + bias

        return prediction

#定义生成器函数
def Generator_RNN (n_samples, charmap_len, BATCH_SIZE,LIMIT_BATCH,seq_len=None, gt=None, rnn_cell=None):

    def get_noise(BATCH_SIZE):#生成随机数
        noise_shape = [BATCH_SIZE, GEN_STATE_SIZE]
        return tf.random_normal(noise_shape,mean =  0.0, stddev=10.0), noise_shape 
    def create_initial_states(noise):
        states = []
        for l in range(GEN_RNN_LAYERS):
            states.append(noise)
        return states     
    
    with tf.variable_scope("Generator"):
        sm_weight = tf.Variable(tf.random_uniform([GEN_STATE_SIZE, charmap_len], minval=-0.1, maxval=0.1))
        sm_bias = tf.Variable(tf.random_uniform([charmap_len], minval=-0.1, maxval=0.1))

        embedding = tf.Variable(tf.random_uniform([charmap_len, GEN_STATE_SIZE], minval=-0.1, maxval=0.1))

        #获得生成器的原始随机数
        char_input = tf.Variable(tf.random_uniform([GEN_STATE_SIZE], minval=-0.1, maxval=0.1))
        #转成一批次的原始随机数据
        char_input = tf.reshape(tf.tile(char_input, [n_samples]), [n_samples, 1, GEN_STATE_SIZE])
        
        cells = []
        for l in range(GEN_RNN_LAYERS):
            cells.append(rnn_cell(GEN_STATE_SIZE))
        if seq_len is None:
            seq_len = tf.placeholder(tf.int32, None, name="ground_truth_sequence_length")

        #初始化rnn的states
        noise, noise_shape = get_noise(BATCH_SIZE)
        train_initial_states = create_initial_states(noise)
        inference_initial_states = create_initial_states(noise)
        if gt is not None: #如果GT不为none,表明当前为训练状态
            train_pred = get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, GEN_STATE_SIZE, seq_len, sm_bias, sm_weight, train_initial_states,BATCH_SIZE,LIMIT_BATCH)
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                GEN_STATE_SIZE, charmap_len, BATCH_SIZE,reuse=True)
        else:#如果GT为None，表面当前为eval状态
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                GEN_STATE_SIZE,charmap_len, BATCH_SIZE,reuse=False)
            train_pred = None

        return train_pred, inference_op

#生成用于训练的模拟样本
def get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len, sm_bias, sm_weight, states,BATCH_SIZE,LIMIT_BATCH):
    gt_embedding = tf.reshape(gt, [n_samples * seq_len, charmap_len])
    gt_RNN_input = gt_embedding@embedding
    gt_RNN_input = tf.reshape(gt_RNN_input, [n_samples, seq_len, num_neurons])[:, :-1]
    gt_sentence_input = tf.concat([char_input, gt_RNN_input], axis=1)#gt_sentence_input的shape[n_samples, seq_len+1, num_neurons]
    RNN_output, _ = rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias, sm_weight, states,BATCH_SIZE)
    train_pred = []
    #从seq_len+1中取出前seq_len个特征，每一个生成的特征都与原来的输入重新组成一个序列
    for i in range(seq_len):
        train_pred.append( #每个序列特征前面加0数据，前i-1行数据，生成的特征最后一个序列数据
            tf.concat([tf.zeros([BATCH_SIZE, seq_len - i - 1, charmap_len]), gt[:, :i], RNN_output[:, i:i + 1, :]],
                      axis=1))

    train_pred = tf.reshape(train_pred, [BATCH_SIZE*seq_len, seq_len, charmap_len])

    if LIMIT_BATCH:#从BATCH_SIZE*seq_len个序列中，随机取出BATCH_SIZE个样本进行判断
        indices = tf.random_uniform([BATCH_SIZE], 0, BATCH_SIZE*seq_len, dtype=tf.int32)#获得随机索引
        train_pred = tf.gather(train_pred, indices)#按照随机索引取数据

    return train_pred

#定义模型函数，通过RNN网络对数据进行特征分析
def rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias, sm_weight, states,BATCH_SIZE
                        ,reuse=False):
    with tf.variable_scope("rnn", reuse=reuse):
        RNN_output = gt_sentence_input
        for l in range(GEN_RNN_LAYERS):
            RNN_output, states[l] = tf.nn.dynamic_rnn(cells[l], RNN_output, dtype=tf.float32,
               initial_state=states[l], scope="layer_%d" % (l + 1))
    RNN_output = tf.reshape(RNN_output, [-1, num_neurons])
    RNN_output = tf.nn.softmax(RNN_output@sm_weight + sm_bias)
    RNN_output = tf.reshape(RNN_output, [BATCH_SIZE, -1, charmap_len])
    return RNN_output, states

#模拟生成真实样本
def get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, states, num_neurons, charmap_len,BATCH_SIZE,
                     reuse=False):
    inference_pred = []
    embedded_pred = [char_input]#以随机生成的第一个为头。后面通过rnn生成序列字符。每个字符通过全链接转成与rnn匹配的向量，再输入rnn
    for i in range(seq_len):
        step_pred, states = rnn_step_prediction(cells, charmap_len, tf.concat(embedded_pred, 1), num_neurons, seq_len,
                                                sm_bias, sm_weight, states,BATCH_SIZE, reuse=reuse)
        best_chars_tensor = tf.argmax(step_pred, axis=2)
        best_chars_one_hot_tensor = tf.one_hot(best_chars_tensor, charmap_len)
        best_char = best_chars_one_hot_tensor[:, -1, :]
        inference_pred.append(tf.expand_dims(best_char, 1))
        embedded_pred.append(tf.expand_dims(best_char@embedding, 1))
        reuse = True 

    return tf.concat(inference_pred, axis=1)

#获得指定训练参数
def params_with_name(name):
    return [p for p in tf.trainable_variables() if name in p.name]

def get_optimization_ops(disc_cost, gen_cost, global_step, gen_lr, disc_lr):
    gen_params = params_with_name('Generator')
    disc_params = params_with_name('Discriminator')
    print("Generator Params: %s" % gen_params)
    print("Disc Params: %s" % disc_params)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=0.5, beta2=0.9).minimize(gen_cost,
        var_list=gen_params,
        global_step=global_step)

    disc_train_op = tf.train.AdamOptimizer(learning_rate=disc_lr, beta1=0.5, beta2=0.9).minimize(disc_cost,
        var_list=disc_params)
    return disc_train_op, gen_train_op

#将输入的序列数据打散。每一个序列作为一个样本
def get_substrings_from_gt(real_inputs, seq_length, charmap_len,BATCH_SIZE,LIMIT_BATCH):
    train_pred = []
    for i in range(seq_length):
        train_pred.append(
            tf.concat([tf.zeros([BATCH_SIZE, seq_length - i - 1, charmap_len]), real_inputs[:, :i + 1]],
                      axis=1))

    all_sub_strings = tf.reshape(train_pred, [BATCH_SIZE * seq_length, seq_length, charmap_len])

    if LIMIT_BATCH:#按照指定批次随机取值
        indices = tf.random_uniform([BATCH_SIZE], 1, all_sub_strings.get_shape()[0], dtype=tf.int32)
        all_sub_strings = tf.gather(all_sub_strings, indices)
        return all_sub_strings[:BATCH_SIZE]
    else:
        return all_sub_strings


def define_objective(charmap, real_inputs_discrete, seq_length, BATCH_SIZE,LIMIT_BATCH):

    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

    train_pred, _ = Generator_RNN(BATCH_SIZE, len(charmap), BATCH_SIZE,LIMIT_BATCH,seq_len=seq_length, gt=real_inputs, rnn_cell=GRUCell)
    
    #将输入real_inputs按照序列展开，再随机取值
    real_inputs_substrings = get_substrings_from_gt(real_inputs, seq_length, len(charmap),BATCH_SIZE,LIMIT_BATCH)

    disc_real = Discriminator_RNN(real_inputs_substrings, len(charmap), seq_length, reuse=False, rnn_cell=GRUCell)
    disc_fake = Discriminator_RNN(train_pred, len(charmap), seq_length, reuse=True, rnn_cell=GRUCell)

    disc_cost, gen_cost = loss_d_g(disc_fake, disc_real, train_pred, real_inputs_substrings, charmap, seq_length, Discriminator_RNN, GRUCell)

    return disc_cost, gen_cost, train_pred, disc_fake, disc_real

#WGAN损失函数 https://arxiv.org/abs/1706.08500
def loss_d_g(disc_fake, disc_real, fake_inputs, real_inputs, charmap, seq_length, Discriminator, GRUCell):
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[tf.shape(real_inputs)[0], 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha * differences)
    gradients = tf.gradients(Discriminator(interpolates, len(charmap), seq_length, reuse=True, rnn_cell=GRUCell), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += LAMBDA * gradient_penalty

    return disc_cost, gen_cost