"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""




# dataset ops
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import *
import numpy as np


###############  range(*args)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.range(5)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(5):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''



###############  zip(datasets)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = Dataset.zip((dataset1,dataset2))
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(5):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''



###############  concatenate(dataset)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = dataset1.concatenate(dataset2)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(10):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''



###############  repeat(count=None)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.repeat(2)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(10):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''


###############  shuffle(buffer_size,seed=None,reshuffle_each_iteration=None)


'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.shuffle(1000)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）

    for i in range(5):		#通过for循环打印所有的数据
        print(sess.run(one_element))				#调用sess.run读出Tensor值

'''




###############  batch(count=None)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.batch(batch_size=2)
iterator = dataset.make_one_shot_iterator()			#从到到尾读一次
one_element = iterator.get_next()					#从iterator里取出一个元素
with tf.Session() as sess:	# 建立会话（session）
	while True:
	    for i in range(2):		#通过for循环打印所有的数据
	        print(sess.run(one_element))				#调用sess.run读出Tensor值
'''

###############  padded_batch

'''
data1 = tf.data.Dataset.from_tensor_slices([[1, 2],[1,3]])
data1 = data1.padded_batch(2,padded_shapes=[4])
iterator = data1.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess2:
    print(sess2.run(init_op))
    print("batched data 1:",sess2.run(next_element))
'''

###############  flat_map(map_func)




'''
import numpy as np

##在内存中生成数据
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = tf.data.Dataset.from_tensor_slices(np.array([[1,2,3],[4,5,6]]))

dataset = dataset.flat_map(lambda x: Dataset.from_tensors(x)) 			
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(10):							#通过for循环打印所有的数据
        print(sess.run(one_element))			#调用sess.run读出Tensor值
'''



######interleave(map_func,cycle_length,block_length=1)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.interleave(lambda x: Dataset.from_tensors(x).repeat(3),
             cycle_length=2, block_length=2)			
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值
'''

######filter(predicate)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.filter(lambda x: tf.less(x, 3))			
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

#过滤掉全为0的元素
dataset = tf.data.Dataset.from_tensor_slices([ [0, 0],[ 3.0, 4.0] ])
dataset = dataset.filter(lambda x: tf.greater(tf.reduce_sum(x), 0))		  #过滤掉全为0的元素	
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

#过滤掉中文字符串(1)加入一个判断列
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha好" ])

def _parse_data(line):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [line], tf.bool)
    #tf.cast(isokstr,tf.bool)[0]

    return line,isokstr#tf.cast(isokstr,tf.bool)[0]
dataset = dataset.map(_parse_data)

dataset = dataset.filter(lambda x,y: y)		  #过滤掉全为0的元素	
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

#过滤掉中文字符串(2)简单实现
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha好" ])

def myfilter(x):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [x], tf.bool)
    return isokstr
dataset = dataset.filter(myfilter)		  #过滤掉全为0的元素	
#dataset = dataset.filter(lambda x,y: y)		  #过滤掉全为0的元素	
iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值

'''
######apply(transformation_func)
'''
data1 = np.arange(50).astype(np.int64)
dataset = tf.data.Dataset.from_tensor_slices(data1)
#将数据集中偶数行与奇数行分开，以window_size为窗口大小，一次取window_size个偶数行和window_size个奇数行。在window_size中，以batch为批次进行分割。
dataset = dataset.apply((tf.contrib.data.group_by_window(key_func=lambda x: x%2, reduce_func=lambda _, els: els.batch(10), window_size=20)  ))

iterator = dataset.make_one_shot_iterator()		#从到到尾读一次
one_element = iterator.get_next()				#从iterator里取出一个元素
with tf.Session() as sess:						#建立会话（session）
    for i in range(100):							#通过for循环打印所有的数据
        print(sess.run(one_element),end=' ')			#调用sess.run读出Tensor值
'''





