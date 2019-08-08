"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import os
import time
import sys
sys.path.append(os.getcwd())


import numpy as np
import tensorflow as tf

model = __import__("10-8  RNNWGAN模型")
dataset = __import__("10-9  mydataset")
mydataset = dataset.mydataset
prepro = __import__("10-10  prepro")
preprosample = prepro.preprosample

#定义单个长度训练的相关参数
CRITIC_ITERS = 2  #每次训练GAN中，迭代训练2次判别器
GEN_ITERS = 10     #每次训练GAN中，迭代训练10次生成器
DISC_LR =2e-4       #定义判别器的学习率
GEN_LR = 1e-4     #定义生成器的学习率

PRINT_ITERATION =100  #定义输出打印信息的迭代频率
SAVE_CHECKPOINTS_EVERY = 1000 #定义保存检查点的迭代频率

LIMIT_BATCH = True  #领生成器生成同批次的数据


#样本数据预处理（用于训练）
def getbacthdata(sess,dosample,next_element,words_redic,BATCH_SIZE,END_SEQ):
    def getone():
        batchx,batchlabel = sess.run(next_element)
        batchx = dosample.ch_to_v([strname.decode() for strname in batchx],words_redic,0)
        batchlabel = np.asarray(batchlabel,np.int32)#no===0  yes==1
        sampletpad ,sampletlengths = dosample.pad_sequences(batchx, maxlen=END_SEQ)#都填充为最大长度END_SEQ
        return sampletpad,batchlabel,sampletlengths

    sampletpad,batchlabel,sampletlengths = getone()
    iii = 0
    while np.shape(sampletpad)[0]!=BATCH_SIZE: #取出不够批次的尾数据
        iii=iii+1
        tf.logging.warn("__________________________iii %d"%iii)
        sampletpad,batchlabel,sampletlengths = getone()

    sampletpad = np.asarray(sampletpad,np.int32)
    return sampletpad,batchlabel,sampletlengths

#获得模拟样本和真实样本（用于测试）
def generate_argmax_samples_and_gt_samples(session, inv_charmap, fake_inputs, disc_fake, _data, real_inputs_discrete, feed_gt=True):
    scores = []
    samples = []
    samples_probabilites = []
    for i in range(10):
        argmax_samples, real_samples, samples_scores = generate_samples(session, inv_charmap, fake_inputs, disc_fake,
                                                                        _data, real_inputs_discrete, feed_gt=feed_gt)
        samples.extend(argmax_samples)
        scores.extend(samples_scores)
        samples_probabilites.extend(real_samples)
    return samples, samples_probabilites, scores

#获得生成的模拟样本
def generate_samples(session, inv_charmap, fake_inputs, disc_fake, _data, real_inputs_discrete, feed_gt=True):
    if feed_gt:
        f_dict = {real_inputs_discrete: _data}
    else:
        f_dict = {}

    fake_samples, fake_scores = session.run([fake_inputs, disc_fake], feed_dict=f_dict)
    fake_scores = np.squeeze(fake_scores)

    decoded_samples = decode_indices_to_string(np.argmax(fake_samples, axis=2), inv_charmap)
    return decoded_samples, fake_samples, fake_scores

#将向量转成字符
def decode_indices_to_string(samples, inv_charmap):
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(inv_charmap[samples[i][j]])

        strde = "".join(decoded)
        decoded_samples.append(strde)
    return decoded_samples

#训练函数
def run(iterations, seq_length, is_first,BATCH_SIZE, prev_seq_length,DATA_DIR,END_SEQ):
    if len(DATA_DIR) == 0:
        raise Exception('Please specify path to data directory in single_length_train.py!')

    dosample = preprosample()
    inv_charmap,charmap = dosample.make_dictionary()

    #获取数据
    next_element = mydataset(DATA_DIR,BATCH_SIZE)

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])

    global_step = tf.Variable(0, trainable=False)
    disc_cost, gen_cost, fake_inputs, disc_fake, disc_real = model.define_objective(charmap, real_inputs_discrete, seq_length,  BATCH_SIZE,LIMIT_BATCH)

    disc_train_op, gen_train_op = model.get_optimization_ops(
        disc_cost, gen_cost, global_step, DISC_LR, GEN_LR)

    saver = tf.train.Saver(tf.trainable_variables())

    config=tf.ConfigProto( log_device_placement=False, allow_soft_placement=True )
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as session:
        checkpoint_dir = './'+str(seq_length)

        session.run(tf.global_variables_initializer())
        if not is_first:
            print("Loading previous checkpoint...")
            internal_checkpoint_dir = './'+str(prev_seq_length)

            kpt = tf.train.latest_checkpoint(internal_checkpoint_dir)
            print("load model:",kpt,internal_checkpoint_dir,seq_length)
            startepo= 0
            if kpt!=None:
                saver.restore(session, kpt)

        _gen_cost_list = []
        _disc_cost_list = []
        _step_time_list = []

        for iteration in range(iterations):
            start_time = time.time()

            #训练判别器
            for i in range(CRITIC_ITERS):
                _data,batchlabel,sampletlengths =getbacthdata(session,dosample,next_element,charmap,BATCH_SIZE,END_SEQ)
                _data= _data[:,:seq_length]
                _disc_cost, _, real_scores = session.run( [disc_cost, disc_train_op, disc_real], feed_dict={real_inputs_discrete: _data} )
                _disc_cost_list.append(_disc_cost)

            #训练生成器
            for i in range(GEN_ITERS):
                _data,batchlabel,sampletlengths =getbacthdata(session,dosample,next_element,charmap,BATCH_SIZE,END_SEQ)
                _data= _data[:,:seq_length]
                _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_inputs_discrete: _data})
                _gen_cost_list.append(_gen_cost)

            _step_time_list.append(time.time() - start_time)

            #显示训练过程中的信息
            if iteration % PRINT_ITERATION == PRINT_ITERATION-1:
                _data,batchlabel,sampletlengths =getbacthdata(session,dosample,next_element,charmap,BATCH_SIZE,END_SEQ)
                _data= _data[:,:seq_length]

                tf.logging.info("iteration %s/%s"%(iteration, iterations))
                tf.logging.info("disc cost {} gen cost {} average step time {}".format( np.mean(_disc_cost_list), np.mean(_gen_cost_list), np.mean(_step_time_list)) )
                _gen_cost_list, _disc_cost_list, _step_time_list = [], [], []

                fake_samples, samples_real_probabilites, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_charmap, fake_inputs, disc_fake, _data, real_inputs_discrete,feed_gt=True)

                print(fake_samples[:2], fake_scores[:2], iteration, seq_length, "train")
                print(decode_indices_to_string(_data[:2], inv_charmap), real_scores[:2], iteration, seq_length, "gt")

            #保持检查点
            if iteration % SAVE_CHECKPOINTS_EVERY == SAVE_CHECKPOINTS_EVERY-1:
                saver.save(session, checkpoint_dir+"/gan.cpkt",   global_step=iteration)

        saver.save(session, checkpoint_dir+"/gan.cpkt",   global_step=iteration)
        session.close()
