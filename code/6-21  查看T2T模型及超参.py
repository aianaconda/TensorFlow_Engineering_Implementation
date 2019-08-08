# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

#6-19  


import tensorflow as tf
from tensor2tensor import models

from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry

print(len(registry.list_models()),registry.list_models())
print(registry.model('transformer'))
print(len(registry.list_hparams()),registry.list_hparams('transformer'))
print(registry.hparams('transformer_base_v1'))