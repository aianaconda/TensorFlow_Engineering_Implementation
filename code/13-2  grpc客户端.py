# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import grpc
import numpy as np
import tensorflow as tf
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

def client_gRPC(data):

    channel = grpc.insecure_channel('127.0.0.1:9000')#建立一个通道
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)#链接远端服务器
     
    #初始化请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'md' #指定模型名称
    request.model_spec.signature_name = "my_signature" #指定模型签名
    request.inputs['input_x'].CopyFrom(tf.contrib.util.make_tensor_proto(data))
    
    #开始调用远端服务。执行预测任务
    start_time = time.time()
    result = stub.Predict(request)
    
    #输出预测时间
    print("花费时间: {}".format(time.time()-start_time))
    
    #解析结果并返回
    result_dict = {}
    for key in result.outputs:
        tensor_proto = result.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        result_dict[key] = nd_array

    return result_dict

def main():
    a = 4.2#传入单个数值
    result= client_gRPC(a)
    print("-------单个数值预测结果-------")
    print(list(result['output']))
    
    #传入多个数值
    data = np.asarray([4.2,4.0],dtype = np.float32)  
    result= client_gRPC(data)
    print("-------多个数值预测结果-------")
    print(list(result['output']))

#主模块运行函数
if __name__ == '__main__':
        main()

