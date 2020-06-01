# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import tensorflow as tf
import numpy as np
import os
tf.compat.v1.disable_v2_behavior()
#在内存中生成模拟数据
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X为-1到1之间连续的100个浮点数
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
    return train_X, train_Y   #以生成器的方式返回

train_data = GenerateData()

#直接使用model定义网络
inputs = tf.keras.Input(shape=(1,))
outputs= tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

#定义2层网络
x = tf.keras.layers.Dense(1, activation='tanh')(inputs)
outputs_2 = tf.keras.layers.Dense(1)(x)
model_2 = tf.keras.Model(inputs=inputs, outputs=outputs_2)



#使用sequential 指定input的形状
model_3 = tf.keras.models.Sequential()
model_3.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model_3.add(tf.keras.layers.Dense(units = 1))



#使用sequential 指定带批次的input形状
model_4 = tf.keras.models.Sequential()
model_4.add(tf.keras.layers.Dense(1, batch_input_shape=(None, 1)))

#使用sequential 指定input的维度
model_5 = tf.keras.models.Sequential()
model_5.add(tf.keras.layers.Dense( 1, input_dim = 1))

#使用sequential 默认input
model_6 = tf.keras.models.Sequential()
model_6.add(tf.keras.layers.Dense(1))
#print(model_6.weights)
model_6.build((None, 1))#指定输入，开始生成模型
#print(model_6.weights)





# 选择损失函数和优化方法
model.compile(loss = 'mse', optimizer = 'sgd')
model_3.compile(loss = tf.compat.v1.losses.mean_squared_error, optimizer = 'sgd')

# 进行训练, 返回损失(代价)函数
for step in range(201):
    cost = model.train_on_batch(train_data[0], train_data[1])
    if step % 10 == 0:
        print ('loss: ', cost)




#直接使用fit函数来训练
model_3.fit(x=train_data[0],y=train_data[1], batch_size=10, epochs=20)

# 获取参数
W,b= model.get_weights()
print ('Weights: ',W)
print ('Biases: ', b)

#对于 使用sequential的模型可以指定具体层来获取
W, b = model_3.layers[0].get_weights()
print ('Weights: ',W)
print ('Biases: ', b)



cost = model.evaluate(train_data[0], train_data[1], batch_size = 10)#测试
print ('test loss: ', cost)

a = model.predict(train_data[0], batch_size = 10)#预测
print(a[:10])
print(train_data[1][:10])


#保存及加载模型
model.save('my_model.h5')

del model  #删除当前模型
#加载
model = tf.keras.models.load_model('my_model.h5')

a = model.predict(train_data[0], batch_size = 10)
print("加载后的测试",a[:10])

#生成tf格式模型
model.save_weights('./keraslog/kerasmodel') #如果是以 '.h5'或'.keras'结尾，默认会生成keras格式模型

#生成tf格式模型，手动指定
os.makedirs("./kerash5log", exist_ok=True)
model.save_weights('./kerash5log/kerash5model',save_format = 'h5')#可以指定save_format为h5 或tf来生成对应的格式


json_string = model.to_json()  #等价于 json_string = model.get_config()
open('my_model.json','w').write(json_string)

#加载模型数据和weights
model_7 = tf.keras.models.model_from_json(open('my_model.json').read())
model_7.load_weights('my_model.h5')
a = model_7.predict(train_data[0], batch_size = 10)
print("加载后的测试",a[:10])

import h5py
f=h5py.File('my_model.h5')
for name in f:
    print(name)
