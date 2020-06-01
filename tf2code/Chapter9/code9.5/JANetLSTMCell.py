import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl 
#from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import random_ops
tf.compat.v1.disable_v2_behavior()

class JANetLSTMCell(kl.LSTMCell):
#class JANetLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=None, #t_max为输入长度的最大值
                 **kwargs):
        self.num_units = num_units
        self.t_max = t_max
        super(JANetLSTMCell, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            all_inputs = tf.concat([x, h], 1)

            num_gates = 2

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh', [x_size, num_gates * self.num_units])
            W_hh = tf.get_variable('W_hh', [self.num_units, num_gates * self.num_units])
            if self.t_max is None:
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=bias_initializer(num_gates))
            else:
                print('Using chrono initializer ...')
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=chrono_init(self.t_max, num_gates))

            weights = tf.concat([W_xh, W_hh], 0)
            concat = tf.nn.bias_add(tf.matmul(all_inputs, weights), bias)
            j, f = tf.split(value=concat, num_or_size_splits=num_gates,axis=1)
            beta = 1
            new_c = tf.sigmoid(f)*c + (1-tf.sigmoid(f-beta))*tf.tanh(j)
            new_h = new_c

            #if self._state_is_tuple:
                #new_state = LSTMStateTuple(new_c, new_h)
            #else:
            new_state = tf.concat([new_c, new_h], 1)
            return new_h, new_state


def chrono_init(t_max, num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        num_units = shape[0]//num_gates
        uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=t_max, dtype=dtype,
                                                    seed=42))
        bias_j = tf.zeros(num_units)
        bias_f = uni_vals
        return tf.concat([bias_j, bias_f], 0)

    return _initializer


def bias_initializer(num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        p = np.zeros(shape)
        num_units = int(shape[0]//num_gates)
        # i, j, o, f
        # f:
        p[-num_units:] = np.ones(num_units)
        return tf.constant(p, dtype)

    return _initializer
