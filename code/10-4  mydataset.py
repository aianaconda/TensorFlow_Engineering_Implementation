"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

class Dataset(object):#定义数据集类。支持动态图和静态图
    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None
        self._is_eager = tf.executing_eagerly()
        self._eager_iterator = None

    def __del__(self): #重载del方法
        if self._sess:#在静态图中，销毁对象时，需要关闭session
            self._sess.close()

    def __iter__(self):#重载迭代器方法
        return self

    def __next__(self):#重载next方法
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):#获取下一个批次的数据
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}): #重置数据集迭代器指针（用于整个数据集循环迭代）
        if self._is_eager:
            self._eager_iterator = tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):#构建数据集
        self._dataset = dataset

        if self._is_eager:#直接返回动态图中的数据集迭代器对象
            self._eager_iterator = tfe.Iterator(dataset)
        else:#在静态图中，需要初始化，并返回迭代器的get_next方法
            self._iterator = dataset.make_initializable_iterator()
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:#如果没有传入session，则需要自己创建一个
                self._sess = tf.Session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self): #返回deatset属性
        return self._dataset

    @property
    def iterator(self):#返回iterator属性
        return self._iterator

    @property
    def batch_op(self):#返回batch_op属性
        return self._batch_op

prefetch_batch=2
num_threads=16
buffer_size=4096

#按照指定的图片目录，读取图片，并转成数据集
def disk_image_batch_dataset(img_paths, batch_size, labels=None, filter=None,drop_remainder=True,
                             map_func=None,  shuffle=True, repeat=-1):

    if labels is None:#将传入的图片路径与标签转成tf.data.Dataset数据集
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    if filter:#支持调用外部传入的filter处理函数
        dataset = dataset.filter(filter)

    def parse_func(path, *label):#定义数据集的map处理函数，读取图片
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3)
        return (img,) + label

    if map_func:#支持调用外部传入的map处理函数
        def map_func_(*args):
            return map_func(*parse_func(*args))
        dataset = dataset.map(map_func_, num_parallel_calls=num_threads)
    else:
        dataset = dataset.map(parse_func, num_parallel_calls=num_threads)

    if shuffle:#乱序操作
        dataset = dataset.shuffle(buffer_size)
    #按批次划分
    dataset = dataset.batch(batch_size,drop_remainder = drop_remainder)
    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)#设置缓存

    return dataset



class Celeba(Dataset):
    #定义人脸属性
    att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    def __init__(self, data_dir, atts, img_resize, batch_size,
                  shuffle=True,  repeat=-1, sess=None, mode='train', crop=True):
        super(Celeba, self).__init__()
        #定义数据路径
        list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
        img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
        img_dir_png = os.path.join(data_dir, 'img_align_celeba_png')

        #读取文本数据
        names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
        if os.path.exists(img_dir_png):#将图片的文件名收集起来
            img_paths = [os.path.join(img_dir_png, name.replace('jpg', 'png')) for name in names]
        elif os.path.exists(img_dir_jpg):
            img_paths = [os.path.join(img_dir_jpg, name) for name in names]
        else:
            raise 'no imgs ! 请先解压配套资源中的样本图片到img_align_celeba里'
        print(img_dir_png,img_dir_jpg)
        #读取每个图片的属性标志
        att_id = [Celeba.att_dict[att] + 1 for att in atts]
        labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int64)

        if img_resize == 64:#将图片剪辑再放大
            offset_h = 40
            offset_w = 15
            img_size = 148
        else:
            offset_h = 26
            offset_w = 3
            img_size = 170

        def _map_func(img, label):
            #从位于(offset_h, offset_w)的图像的左上角像素开始，对图像裁剪
            img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            #使用双向插值法缩放图片
            img = tf.image.resize(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1#归一化
            label = (label + 1) // 2#将标签变为0和1
            return img, label

        drop_remainder = True
        if mode == 'test':#根据使用情况，决定数据集的处理
            drop_remainder = False
            shuffle = False
            repeat = 1
            img_paths = img_paths[182637:]
            labels = labels[182637:]
        elif mode == 'val':
            img_paths = img_paths[182000:182637]
            labels = labels[182000:182637]
        else:
            img_paths = img_paths[:182000]
            labels = labels[:182000]
        #创建数据集
        dataset = disk_image_batch_dataset(img_paths=img_paths,labels=labels,
                                           batch_size=batch_size, map_func=_map_func,
                                           drop_remainder=drop_remainder,
                                           shuffle=shuffle,repeat=repeat)
        self._bulid(dataset, sess)#构建数据集
        self._img_num = len(img_paths)#计算总长度

    def __len__(self):#重载len函数
        return self._img_num#返回数据集的总长度

    @staticmethod#定义一个静态方法，实现将冲突类别清0
    def check_attribute_conflict(att_batch, att_name, att_names):
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        att_id = att_names.index(att_name)
        for att in att_batch:#循环处理批次中的每个反向标签
            if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] == 1:
                _set(att, 0, 'Bangs')#没头发和退缩发际线与头帘冲突
            elif att_name == 'Bangs' and att[att_id] == 1:
                _set(att, 0, 'Bald')
                _set(att, 0, 'Receding_Hairline')
            elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] == 1:
                for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    if n != att_name: #头发颜色只能取一种
                        _set(att, 0, n)
            elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] == 1:
                for n in ['Straight_Hair', 'Wavy_Hair']:
                    if n != att_name:#直发和波浪
                        _set(att, 0, n)
            elif att_name in ['Mustache', 'No_Beard'] and att[att_id] == 1:
                for n in ['Mustache', 'No_Beard']:#有胡子和没胡子
                    if n != att_name:
                        _set(att, 0, n)

        return att_batch



