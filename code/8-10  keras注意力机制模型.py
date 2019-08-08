#! -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K #载入keras的后端实现
 
class Position_Embedding(keras.layers.Layer):    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        position_j = 1. / K.pow(  10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size  )
        position_j = K.expand_dims(position_j, 0)
        #按照x的1维度累计求和，与arange一样，生成序列。只不过按照x的实际长度来
        position_i = tf.cumsum(K.ones_like(x[:,:,0]), 1)-1 
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


class Attention(keras.layers.Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head#输出的总维度
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(int(input_shape[0][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(int(input_shape[1][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(int(input_shape[2][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
  
    def Mask(self, inputs, seq_len, mode='mul'):#按照seq_len实际长度对inputs进行计算
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        
        if len(x) == 3:#解析传入的入Q_seq,K_seq,V_seq
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:#Q_len,V_len为mask的长度
            Q_seq,K_seq,V_seq,Q_len,V_len = x
            
        print("Q_seq",Q_seq)
        #对Q、K、V做线性变换,一共做nb_head次，每次线性变化成size_per_head维度
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))#相当于transpose,排列各维度的顺序
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
        


class TargetedDropout(keras.layers.Layer): #Targeted Dropout层

    def __init__(self,drop_rate,target_rate, **kwargs):

        super(TargetedDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.drop_rate = drop_rate
        self.target_rate = target_rate

    def get_config(self):
        config = {
            'drop_rate': self.drop_rate,
            'target_rate': self.target_rate,
        }
        base_config = super(TargetedDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _compute_target_mask(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        input_type = K.dtype(inputs)

        mask_threshold = K.constant(1e8, dtype=input_type)
        
        channel_num = int(inputs.shape[-1])
        channel_dim = K.prod(input_shape[:-1])
        masked_inputs = inputs
        if mask is not None:
            masked_inputs = K.switch(
                K.cast(mask, K.floatx()) > 0.5,
                masked_inputs,
                K.ones_like(masked_inputs, dtype=input_type) * mask_threshold
            )
        norm = K.abs(masked_inputs)
        channeled_norm = K.transpose(K.reshape(norm, (channel_dim, channel_num)))
        weight_num = K.sum(
            K.reshape(K.cast(masked_inputs < mask_threshold, K.floatx()), (channel_dim, channel_num)),
            axis=0,
        )
        indices = K.stack(
            [
                K.arange(channel_num, dtype='int32'),
                K.cast(self.target_rate * weight_num, dtype='int32') - 1,
            ],
            axis=-1,
        )
        threshold = -tf.gather_nd(tf.nn.top_k(-channeled_norm, k=K.max(indices[:, 1]) + 1).values, indices)
        
        threshold = K.reshape(tf.tile(threshold, [channel_dim]), input_shape)
        target_mask = K.switch(
            norm <= threshold,
            K.ones_like(inputs, dtype=K.floatx()),
            K.zeros_like(inputs, dtype=K.floatx()),
        )
        return target_mask

    def call(self, inputs, mask=None, training=None):
        target_mask = self._compute_target_mask(inputs, mask=mask)

        def dropped_mask():
            drop_mask = K.switch(
                K.random_uniform(K.shape(inputs)) < self.drop_rate,
                K.ones_like(inputs, K.floatx()),
                K.zeros_like(inputs, K.floatx()),
            )
            return target_mask * drop_mask

        def pruned_mask():
            return target_mask

        mask = K.in_train_phase(dropped_mask, pruned_mask, training=training)
        outputs = K.switch(
            mask > 0.5,
            K.zeros_like(inputs, dtype=K.dtype(inputs)),
            inputs,
        )
        return outputs
