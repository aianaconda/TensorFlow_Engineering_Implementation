"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from switchnorm import SwitchNormalization

ngf = 64  #定义生成器原始卷积核个数
ndf = 64#定义判别器原始卷积核个数
input_nc = 3#定义输入通道
output_nc = 3#定义输出通道

n_blocks_gen = 9#定义残差层数量


#定义残差块函数
def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):

    x = KL.Conv2D(filters=filters, #使用步长为1的卷积，保持大小不变
               kernel_size=kernel_size,
               strides=strides, padding='same')(input)    

    x = SwitchNormalization()(x)
    x = KL.Activation('relu')(x)

    if use_dropout:         #使用dropout
        x = KL.Dropout(0.5)(x)
    
    x = KL.Conv2D(filters=filters,  #再来一次步长为1的卷积
               kernel_size=kernel_size,
               strides=strides,padding='same')(x)    
    

    x = SwitchNormalization()(x)
    
    #将卷积后的结果与原始输入相加
    merged = KL.Add()([input, x])#残差层
    return merged




def generator_model(image_shape,istrain = True):#构建生成器模型

    #构建输入层（与动态图不兼容）
    inputs = KL.Input(shape=(image_shape[0],image_shape[1], input_nc))
    #步长为1的卷积操作，尺寸不变
    x = KL.Conv2D(filters=ngf, kernel_size=(7, 7), padding='same')(inputs)

    x = SwitchNormalization()(x)
    x = KL.Activation('relu')(x)

    n_downsampling = 2
    for i in range(n_downsampling):#两次下采样
        mult = 2**i
        x = KL.Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = SwitchNormalization()(x)
        x = KL.Activation('relu')(x)

    mult = 2**n_downsampling
    for i in range(n_blocks_gen):#定义多个残差层
        x = res_block(x, ngf*mult, use_dropout=istrain)

    for i in range(n_downsampling):#两次上采样
        mult = 2**(n_downsampling - i)
        x = KL.UpSampling2D()(x)
        x = KL.Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
        x = SwitchNormalization()(x)
        x = KL.Activation('relu')(x)

    #步长为1的卷积操作
    x = KL.Conv2D(filters=output_nc, kernel_size=(7, 7), padding='same')(x)
    x = KL.Activation('tanh')(x)

    outputs = KL.Add()([x, inputs])#与最外层的输入完成一次大残差
    #防止特征值域过大，进行除2操作（取平均数残差）
    outputs = KL.Lambda(lambda z: z/2)(outputs)
    #构建模型
    model = KM.Model(inputs=inputs, outputs=outputs, name='Generator')
    return model


def discriminator_model(image_shape):#构建判别器模型

    n_layers, use_sigmoid = 3, False
    inputs = KL.Input(shape=(image_shape[0],image_shape[1],output_nc))
    #下采样卷积
    x = KL.Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
    x = KL.LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):#继续3次下采样卷积
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = KL.Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
        x = KL.BatchNormalization()(x)
        x = KL.LeakyReLU(0.2)(x)

    #步长为1的卷积操作，尺寸不变
    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = KL.Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = KL.BatchNormalization()(x)
    x = KL.LeakyReLU(0.2)(x)
    
    #步长为1的卷积操作，尺寸不变。将通道压缩为1
    x = KL.Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = KL.Activation('sigmoid')(x)

    x = KL.Flatten()(x) #两层全连接，输出判别结果
    x = KL.Dense(1024, activation='tanh')(x)
    x = KL.Dense(1, activation='sigmoid')(x)

    model = KM.Model(inputs=inputs, outputs=x, name='Discriminator')
    return model

#将判别器与生成器结合起来，构成完整模型
def g_containing_d_multiple_outputs(generator, discriminator,image_shape):
    inputs = KL.Input(shape=(image_shape[0],image_shape[1],input_nc)  )
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = KM.Model(inputs=inputs, outputs=[generated_image, outputs])
    return model



