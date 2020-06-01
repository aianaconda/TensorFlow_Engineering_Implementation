"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

from functools import partial
import traceback
import re
import numpy as np
import tensorflow as tf
import time
import os
import scipy.misc
mydataset = __import__("10-4  mydataset")
data = mydataset#.data

AttGANmodels = __import__("10-5  AttGANmodels")
models = AttGANmodels#.models

img_size = 128
#定义模型参数
shortcut_layers = 1
inject_layers =1
enc_dim = 64
dec_dim = 64
dis_dim = 64
dis_fc_dim = 1024
enc_layers = 5
dec_layers = 5
dis_layers = 5

#定义训练参数
mode = 'wgan'#还可以设置为'lsgan'
epoch = 200
batch_size = 32
lr_base = 0.0002
n_d = 5     #执行n_d次判别器 一次生成器
b_distribution = 'none'#定义生成器的随机方式 'none', 'uniform', 'truncated_normal'
thres_int = 0.5 #训练时，特征的上下限值域
test_int = 1.0#测试时，特征属性的上下限值域（一般要大于训练时的值域，使特征更加明显）
n_sample = 32

#定义默认属性
att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
n_att = len(att_default)

experiment_name = "128_shortcut1_inject1_None"#定义模型文件夹名称
#创建目录
os.makedirs('./output/%s' % experiment_name, exist_ok=True)


tf.reset_default_graph()
#定义运行session的硬件配置
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#建立数据集
tr_data = data.Celeba(r'.\data', att_default, img_size, batch_size, mode='train', sess=sess)
val_data = data.Celeba(r'.\data', att_default, img_size, n_sample, mode='val', shuffle=False, sess=sess)
#准备一部分评估样本，用与看模型的输出效果
val_data.get_next()
val_data.get_next()
xa_sample_ipt, a_sample_ipt = val_data.get_next()
b_sample_ipt_list = [a_sample_ipt]  #保存原始样本标签，用于重建
for i in range(len(att_default)): #每个属性生成一个标签
    tmp = np.array(a_sample_ipt, copy=True)
    tmp[:, i] = 1 - tmp[:, i]   #将指定属性取反。显像的属性去掉冲突项
    tmp = data.Celeba.check_attribute_conflict(tmp, att_default[i], att_default)
    b_sample_ipt_list.append(tmp)



#构建模型
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)
D = partial(models.D, n_att=n_att, dim=dis_dim, fc_dim=dis_fc_dim, n_layers=dis_layers)

#定义学习率占位符
lr = tf.placeholder(dtype=tf.float32, shape=[])

xa = tr_data.batch_op[0]
a = tr_data.batch_op[1]
#将标签值域，由0、1变为-0.5、0.5
_a = (tf.cast(a,tf.float32) * 2 - 1) * thres_int
b = tf.random_shuffle(a)#打乱属性标签的对应关系，用于生成器的输入
if b_distribution == 'none':
    _b = (tf.cast(b,tf.float32) * 2 - 1) * thres_int
elif b_distribution == 'uniform':
    _b = (tf.cast(b,tf.float32) * 2 - 1) * tf.random_uniform(tf.shape(b)) * (2 * thres_int)
elif b_distribution == 'truncated_normal':
    _b = (tf.cast(b,tf.float32) * 2 - 1) * (tf.truncated_normal(tf.shape(b)) + 2) / 4.0 * (2 * thres_int)

xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# generate
z = Genc(xa) #压缩特征
xb_ = Gdec(z, _b) #将压缩特征配合指定属性，生成人脸图片（用于对抗）
with tf.control_dependencies([xb_]):
    xa_ = Gdec(z, _a)#将压缩特征配合原有标签属性，生成人脸图片（用于重建）

# discriminate
xa_logit_gan, xa_logit_att = D(xa)
xb__logit_gan, xb__logit_att = D(xb_)

# discriminator losses
if mode == 'wgan':  # wgan-gp
    wd = tf.reduce_mean(xa_logit_gan) - tf.reduce_mean(xb__logit_gan)
    d_loss_gan = -wd
    gp = models.gradient_penalty(D, xa, xb_)
elif mode == 'lsgan':  # lsgan-gp
    xa_gan_loss = tf.losses.mean_squared_error(tf.ones_like(xa_logit_gan), xa_logit_gan)
    xb__gan_loss = tf.losses.mean_squared_error(tf.zeros_like(xb__logit_gan), xb__logit_gan)
    d_loss_gan = xa_gan_loss + xb__gan_loss
    gp = models.gradient_penalty(D, xa)


xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)#计算分类器的重建损失
d_loss = d_loss_gan + gp * 10.0 + xa_loss_att

# generator losses
if mode == 'wgan':
    xb__loss_gan = -tf.reduce_mean(xb__logit_gan)
elif mode == 'lsgan':
    xb__loss_gan = tf.losses.mean_squared_error(tf.ones_like(xb__logit_gan), xb__logit_gan)


xb__loss_att = tf.losses.sigmoid_cross_entropy(b, xb__logit_att)#计算分类器的重建损失
xa__loss_rec = tf.losses.absolute_difference(xa, xa_)#用于校准生成器按照属性生成的结果


g_loss = xb__loss_gan + xb__loss_att * 10.0 + xa__loss_rec * 100.0

#获得训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'D' in var.name]
g_vars = [var for var in t_vars if 'G' in var.name]
#定义优化器OP
d_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_vars)

#
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)

def summary(tensor_collection,
            summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram'],
            scope=None):

    def _summary(tensor, name, summary_type):
        if name is None:
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        summaries = []
        if len(tensor.shape) == 0:
            summaries.append(tf.summary.scalar(name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(name, tensor))
        return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]

    with tf.name_scope(scope, 'summary'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)

# summary
d_summary = summary({d_loss_gan: 'd_loss_gan',gp: 'gp',
    xa_loss_att: 'xa_loss_att',}, scope='D')

lr_summary = summary({lr: 'lr'}, scope='Learning_Rate')

g_summary = summary({ xb__loss_gan: 'xb__loss_gan',
    xb__loss_att: 'xb__loss_att',xa__loss_rec: 'xa__loss_rec',
}, scope='G')

d_summary = tf.summary.merge([d_summary, lr_summary])





def counter(start=0, scope=None):#对张量进行计数
    with tf.variable_scope(scope, 'counter'):
        counter = tf.get_variable(name='counter',
                                  initializer=tf.constant_initializer(start),
                                  shape=(),
                                  dtype=tf.int64)
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt
#定义计数器
it_cnt, update_cnt = counter()

#定义saver用于读取模型
saver = tf.train.Saver(max_to_keep=1)

#定义摘要日志写入器
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)





def immerge(images, row, col):#合成图片
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img

#转化图片值域，从[-1.0, 1.0] 到 [min_value, max_value]
def to_range(images, min_value=0.0, max_value=1.0, dtype=None):

    assert np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        ('The input images should be float64(32) '
         'and in the range of [-1.0, 1.0]!')
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) +
            min_value).astype(dtype)

def imwrite(image, path):#保存图片 [-1.0, 1.0]
    if image.ndim == 3 and image.shape[2] == 1:  #保存灰度图
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))




#创建或加载模型
ckpt_dir = './output/%s/checkpoints' % experiment_name
try:
    thisckpt_dir = tf.train.latest_checkpoint(ckpt_dir)
    restorer = tf.train.Saver()
    restorer.restore(sess, thisckpt_dir)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % thisckpt_dir)
except:
    print(' [*] No checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)
    sess.run(tf.global_variables_initializer())

#训练模型
try:

    #计算训练一次数据集所需的迭代次数
    it_per_epoch = len(tr_data) // (batch_size * (n_d + 1))
    max_it = epoch * it_per_epoch
    for it in range(sess.run(it_cnt), max_it):
        start_time = time.time()
        sess.run(update_cnt)#更新计数器

        #计算训练一次数据集所需要的迭代次数
        epoch = it // it_per_epoch
        it_in_epoch = it % it_per_epoch + 1

        #计算学习率
        lr_ipt = lr_base / (10 ** (epoch // 100))

        #训练判别器
        for i in range(n_d):
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        #训练生成器
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={lr: lr_ipt})
        summary_writer.add_summary(g_summary_opt, it)

        #显示计算时间
        if (it + 1) % 1 == 0:
            print("Epoch: {} {}/{} time: {}".format(epoch, it_in_epoch, it_per_epoch,time.time()-start_time))

        #保存模型
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
            print('Model is saved at %s!' % save_path)

        #用模型生成一部分样本。用于观察效果
        if (it + 1) % 100 == 0:
            x_sample_opt_list = [xa_sample_ipt, np.full((n_sample, img_size, img_size // 10, 3), -1.0)]
            for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int#标签预处理
                if i > 0:   ##将需要变化的那列属性值域调成[-1，1]当 i为 0时，是原始标签
                    _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
                x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
            sample = np.concatenate(x_sample_opt_list, 2)

            save_dir = './output/%s/sample_training' % experiment_name

            os.makedirs(save_dir, exist_ok=True)
            imwrite(immerge(sample, n_sample, 1), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_in_epoch, it_per_epoch))

except:
    traceback.print_exc()
finally:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
    print('Model is saved at %s!' % save_path)
    sess.close()
