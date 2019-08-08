"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import numpy as np
from PIL import Image
import glob
import os
import tensorflow as tf

deblurmodel = __import__("10-1  deblurmodel")
generator_model = deblurmodel.generator_model

def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


batch_size = 4
#RESHAPE = (256,256) #定义处理图片的大小
RESHAPE = (360,640) #定义处理图片的大小

path = r'./image/test'
A_paths, B_paths = os.path.join(path, 'A', "*.png"), os.path.join(path, 'B', "*.png")
#获取该路径下的png文件
A_fnames, B_fnames = glob.glob(A_paths),glob.glob(B_paths)
#生成Dataset对象
dataset = tf.data.Dataset.from_tensor_slices((A_fnames, B_fnames))

def _processimg(imgname):#定义函数调整图片大小
    image_string = tf.read_file(imgname)         		#读取整个文件
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

dataset = dataset.map(_parseone)   				#转化为有图片内容的数据集
dataset = dataset.batch(batch_size)             #将数据集按照batch_size划分
dataset = dataset.prefetch(1)

#生成数据集迭代器
iterator = dataset.make_initializable_iterator()
datatensor = iterator.get_next()

g = generator_model(RESHAPE,False)
g.load_weights("generator_499_0.h5")

#定义配置文件
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)#建立session
sess.run( iterator.initializer )
ii= 0
while True:
    try:#获取一批次的数据
        (x_test,y_test) = sess.run(datatensor)
    except tf.errors.OutOfRangeError:
        break #如果数据取完则退出循环
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)
    print(generated_images.shape[0])
    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
#        im = im.resize( (640*3, int( 640*720/1280)   ) )
        print('results{}{}.png'.format(ii,i))
        im.save('results{}{}.png'.format(ii,i))
        im2 = Image.fromarray(img.astype(np.uint8))
#        im2 = im2.resize( (640, int( 640*720/1280)   ) )
        im2.save('results2{}{}.png'.format(ii,i))
    ii+=1


