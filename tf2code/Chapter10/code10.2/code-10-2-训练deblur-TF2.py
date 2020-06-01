"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import os
import datetime
import numpy as np
import tqdm
import tensorflow as tf
import glob
from tensorflow.python.keras.applications.vgg16 import VGG16
from functools import partial
from tensorflow.keras import models as KM
from tensorflow.keras import backend as K #载入keras的后端实现

tf.compat.v1.disable_v2_behavior()
deblurmodel = __import__("code-10-1-deblurmodel-TF2")
generator_model = deblurmodel.generator_model
discriminator_model = deblurmodel.discriminator_model
g_containing_d_multiple_outputs = deblurmodel.g_containing_d_multiple_outputs

RESHAPE = (256,256) #定义处理图片的大小
#RESHAPE = (360,640) #
epoch_num = 500     #定义迭代训练次数

batch_size =2    #定义批次大小
critic_updates = 5  #定义每训练一次生成器，所需要训练判别器的次数
#保存模型
BASE_DIR = 'weights/'
def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    os.makedirs(save_dir, exist_ok=True)#创建目录
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


path = r'./image/train'
A_paths, B_paths = os.path.join(path, 'A', "*.png"), os.path.join(path, 'B', "*.png")
#获取该路径下的png文件
A_fnames, B_fnames = glob.glob(A_paths),glob.glob(B_paths)
#生成Dataset对象
dataset = tf.data.Dataset.from_tensor_slices((A_fnames, B_fnames))

print(A_fnames[0])

def _processimg(imgname):#定义函数调整图片大小
    image_string = tf.io.read_file(imgname)         		#读取整个文件
    image_decoded = tf.image.decode_image(image_string)
    image_decoded.set_shape([None, None, None])#形状变化，不然下面会转化失败
    #变化尺寸
    img =tf.image.resize( image_decoded,RESHAPE)#[RESHAPE[0],RESHAPE[1],3])
    image_decoded = (img - 127.5) / 127.5
    return image_decoded

def _parseone(A_fname, B_fname):  	            		#解析一个图片文件
    #读取并预处理图片
    image_A,image_B = _processimg(A_fname),_processimg(B_fname)
    return image_A,image_B

dataset = dataset.shuffle(buffer_size=len(B_fnames))
dataset = dataset.map(_parseone)   				#转化为有图片内容的数据集
#dataset = dataset.repeat(epoch_num)            #将数据集重复num_epochs次
dataset = dataset.batch(batch_size)             #将数据集按照batch_size划分
dataset = dataset.prefetch(1)


#获得模型
g = generator_model(RESHAPE) #生成器模型
d = discriminator_model(RESHAPE)#判别器模型
d_on_g = g_containing_d_multiple_outputs(g, d,RESHAPE)#联合模型

#定义优化器
d_opt = tf.keras.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
d_on_g_opt = tf.keras.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



#WGAN的损失
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(input_tensor=y_true*y_pred)

d.trainable = True
d.compile(optimizer=d_opt, loss=wasserstein_loss)#编译模型
d.trainable = False

#计算特征空间损失
def perceptual_loss(y_true, y_pred,image_shape):
    vgg = VGG16(include_top=False, weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                input_shape=(image_shape[0],image_shape[1],3) )

    loss_model = KM.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return tf.reduce_mean(input_tensor=tf.square(loss_model(y_true) - loss_model(y_pred)))

myperceptual_loss = partial(perceptual_loss, image_shape=RESHAPE)
myperceptual_loss.__name__ = 'myperceptual_loss'

#构建损失
loss = [myperceptual_loss, wasserstein_loss]
loss_weights = [100, 1]#将损失调为统一数量级
d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
d.trainable = True

output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

#生成数据集迭代器
iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
datatensor = iterator.get_next()

#定义配置文件
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.compat.v1.Session(config=config)#建立session

def pre_train_epoch(sess, iterator,datatensor):#迭代整个数据集进行训练
    d_losses = []
    d_on_g_losses = []
    sess.run( iterator.initializer )

    while True:
        try:#获取一批次的数据
            (image_blur_batch,image_full_batch) = sess.run(datatensor)
        except tf.errors.OutOfRangeError:
            break #如果数据取完则退出循环

        #将模糊图片输入生成器
        generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

        for _ in range(critic_updates):#训练5次判别器
            #训练，并计算真样本的loss
            d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
            #训练，并计算模拟样本的loss
            d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)#二者相加，再除2
            d_losses.append(d_loss)

        d.trainable = False#固定判别器参数
        #训练并计算生成器loss。
        d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
        d_on_g_losses.append(d_on_g_loss)

        d.trainable = True#恢复判别器参数可训练的属性
        if len(d_on_g_losses)%10== 0:
            print(len(d_on_g_losses),np.mean(d_losses), np.mean(d_on_g_losses))
    return np.mean(d_losses), np.mean(d_on_g_losses)

tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
for epoch in tqdm.tqdm(range(epoch_num)):#按照指定次数迭代训练

    #迭代训练一次数据集
    dloss,gloss = pre_train_epoch(sess, iterator,datatensor)
    with open('log.txt', 'a+') as f:
        f.write('{} - {} - {}\n'.format(epoch, dloss, gloss))
    #保存模型
    save_all_weights(d, g, epoch, int(gloss))

sess.close()



