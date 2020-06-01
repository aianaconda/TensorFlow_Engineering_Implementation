"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import os
import tensorflow as tf

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def load_sample(sample_dir,shuffleflag = True):
    '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
    print ('loading sample  dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#递归遍历文件夹
        for filename in filenames:                            #遍历所有文件名
            #print(dirnames)
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #添加文件名
            labelsnames.append( dirpath.split('\\')[-1] )#添加文件名对应的标签

    lab= list(sorted(set(labelsnames)))  #生成标签名称列表
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #生成字典

    labels = [labdict[i] for i in labelsnames]
    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)



directory='man_woman\\'                                                     #定义样本路径
(filenames,labels),_ =load_sample(directory,shuffleflag=False) #载入文件名称与标签


def _distorted_image(image,size,ch=1,shuffleflag = False,cropflag  = False,
                     brightnessflag=False,contrastflag=False):    #定义函数，实现变化图片
    distorted_image =tf.image.random_flip_left_right(image)

    if cropflag == True:                                                #随机裁剪
        s = tf.random_uniform((1,2),int(size[0]*0.8),size[0],tf.int32)
        distorted_image = tf.random_crop(distorted_image, [s[0][0],s[0][0],ch])

    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转
    if brightnessflag == True:#随机变化亮度
        distorted_image = tf.image.random_brightness(distorted_image,max_delta=10)
    if contrastflag == True:   #随机变化对比度
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
    if shuffleflag==True:
        distorted_image = tf.random_shuffle(distorted_image)#沿着第0维乱序
    return distorted_image


def _norm_image(image,size,ch=1,flattenflag = False):    #定义函数，实现归一化，并且拍平
    image_decoded = image/255.0
    if flattenflag==True:
        image_decoded = tf.reshape(image_decoded, [size[0]*size[1]*ch])
    return image_decoded

from skimage import transform
def _random_rotated30(image, label): #定义函数实现图片随机旋转操作
    
    def _rotated(image):                #封装好的skimage模块，来进行图片旋转30度
        shift_y, shift_x = np.array(image.shape.as_list()[:2],np.float32) / 2.
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv,image.size = transform.SimilarityTransform(translation=[shift_x, shift_y]),image.shape#兼容transform函数
        image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
        return image_rotated

    def _rotatedwrap():
        image_rotated = tf.py_function( _rotated,[image],[tf.float64])   #调用第三方函数
        return tf.cast(image_rotated,tf.float32)[0]

    a = tf.random_uniform([1],0,2,tf.int32)#实现随机功能
    image_decoded = tf.cond(tf.equal(tf.constant(0),a[0]),lambda: image,_rotatedwrap)

    return image_decoded, label



def dataset(directory,size,batchsize,random_rotated=False):#定义函数，创建数据集
    """ parse  dataset."""
    (filenames,labels),_ =load_sample(directory,shuffleflag=False) #载入文件名称与标签
    def _parseone(filename, label):                         #解析一个图片文件
        """ Reading and handle  image"""
        image_string = tf.read_file(filename)         #读取整个文件
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])    # 必须有这句，不然下面会转化失败
        image_decoded = _distorted_image(image_decoded,size)#对图片做扭曲变化
        image_decoded = tf.image.resize(image_decoded, size)  #变化尺寸
        image_decoded = _norm_image(image_decoded,size)#归一化
        image_decoded = tf.cast(image_decoded,dtype=tf.float32)
        label = tf.cast(  tf.reshape(label, []) ,dtype=tf.int32  )#将label 转为张量
        return image_decoded, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))#生成Dataset对象
    dataset = dataset.map(_parseone)   #有图片内容的数据集

    if random_rotated == True:
        dataset = dataset.map(_random_rotated30)

    dataset = dataset.batch(batchsize) #批次划分数据集

    return dataset




#如果显示有错，可以尝试使用np.reshape(thisimg, (size[0],size[1],3))或
#np.asarray(thisimg[0], dtype='uint8')改变类型与形状
def showresult(subplot,title,thisimg):          #显示单个图片
    p =plt.subplot(subplot)
    p.axis('off')
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #显示
    plt.figure(figsize=(20,10))     #定义显示图片的宽、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素
    return one_element

sample_dir=r"man_woman"
size = [96,96]
batchsize = 10
tdataset = dataset(sample_dir,size,batchsize)
tdataset2 = dataset(sample_dir,size,batchsize,True)
print(tdataset.output_types)  #打印数据集的输出信息
print(tdataset.output_shapes)

one_element1 = getone(tdataset)				#从tdataset里取出一个元素
one_element2 = getone(tdataset2)				#从tdataset2里取出一个元素


with tf.Session() as sess:	# 建立会话（session）
    sess.run(tf.global_variables_initializer())  #初始化

    try:
        for step in np.arange(1):
            value = sess.run(one_element1)
            value2 = sess.run(one_element2)

            showimg(step,value[1],np.asarray( value[0]*255,np.uint8),10)       #显示图片
            showimg(step,value2[1],np.asarray( value2[0]*255,np.uint8),10)       #显示图片


    except tf.errors.OutOfRangeError:           #捕获异常
        print("Done!!!")



