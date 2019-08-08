# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""

import os
import pandas as pd
import six
import tensorflow as tf
import tensorflow_lattice as tfl

#定义数据集目录.
testdir = "./income_data/adult.test.csv.txt"
traindir = "./income_data/adult.data.csv.txt"

batch_size = 1000 #定义批次

#定义列名，对应csv中的数据列.
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt",
    "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_area",
    "income_bracket"
]


_df_data = {}       #以字典形式，存放csv文件名和对应的样本内容
_df_data_labels = {} #以字典形式，存放csv文件名和对应的标签内容

#读入原始csv文件，并转成估算器的输入函数
def get_input_fn(file_path, batch_size, num_epochs, shuffle):

  if file_path not in _df_data: #保证只读取依次csv
    #读取csv文件，并将样本内容放入_df_data中
    _df_data[file_path] = pd.read_csv( tf.gfile.Open(file_path),
        names=CSV_COLUMNS,skipinitialspace=True,
        engine="python", skiprows=1)
    
    _df_data[file_path] = _df_data[file_path].dropna(how="any", axis=0)
    #读取csv文件，并将标签内容放入_df_data_labels中
    _df_data_labels[file_path] = _df_data[file_path]["income_bracket"].apply(
        lambda x: ">50K" in x).astype(int)
    
  return tf.estimator.inputs.pandas_input_fn( #返回pandas结构的输入函数
      x=_df_data[file_path],y=_df_data_labels[file_path],
      batch_size=batch_size,shuffle=shuffle,
      num_epochs=num_epochs,num_threads=1)

def create_feature_columns():#创建特征列
  #离散列.
  gender = tf.feature_column.categorical_column_with_vocabulary_list(
      "gender", ["Female", "Male"])
  education = tf.feature_column.categorical_column_with_vocabulary_list(
      "education", [
          "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
          "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",
          "5th-6th", "10th", "1st-4th", "Preschool", "12th"
      ])
  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      "marital_status", [
          "Married-civ-spouse", "Divorced", "Married-spouse-absent",
          "Never-married", "Separated", "Married-AF-spouse", "Widowed"
      ])
  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      "relationship", [
          "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
          "Other-relative"
      ])
  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      "workclass", [
          "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
          "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
      ])
  occupation = tf.feature_column.categorical_column_with_vocabulary_list(
      "occupation", [
          "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
          "Sales", "Other-service", "Machine-op-inspct", "?",
          "Transport-moving", "Handlers-cleaners", "Farming-fishing",
          "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"
      ])
  race = tf.feature_column.categorical_column_with_vocabulary_list(
      "race", [ "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
               "Other",]   )
  native_area = tf.feature_column.categorical_column_with_vocabulary_list(
      "native_area", ["area_A","area_B","?", "area_C",
          "area_D", "area_E", "area_F","area_G","area_H","area_I",
          "Greece", "area_K","area_L","area_M","area_N","area_O",
          "area_P","Italy","area_R", "Jamaica","area_T","Mexico","area_S",
          "area_U","France","area_W","area_V","Ecuador","area_X", "Columbia",
          "area_Y", "Guatemala","Nicaragua","area_Z", "area_1A",
          "area_1B", "area_1C","area_1D","Peru",
          "area_#", "area_1G",])

  #连续值列.
  age = tf.feature_column.numeric_column("age")
  education_num = tf.feature_column.numeric_column("education_num")
  capital_gain = tf.feature_column.numeric_column("capital_gain")
  capital_loss = tf.feature_column.numeric_column("capital_loss")
  hours_per_week = tf.feature_column.numeric_column("hours_per_week")
  
  #将处理好的特征列返回。fnlwgt不需要，income-bracket是标签，也不需要
  return [ age, workclass, education, education_num, marital_status,
      occupation, relationship,race,gender, capital_gain,
      capital_loss, hours_per_week, native_area,]


#创建校准关键点
def create_quantiles(quantiles_dir):
  batch_size = 10000 #设置批次
  
  #创建输入函数
  input_fn =get_input_fn(traindir, batch_size, num_epochs=1, shuffle=False)
  
  tfl.save_quantiles_for_keypoints( #默认保存1000个校准关键点
      input_fn=input_fn,
      save_dir=quantiles_dir, #默认会建立一个文件目录
      feature_columns=create_feature_columns(),
      num_steps=None)
  

quantiles_dir = "./"
create_quantiles(quantiles_dir) #创建校准关键点信息
a = tfl.load_keypoints_from_quantiles(["age"],quantiles_dir,num_keypoints=10,
                                  output_min=17.0,output_max=90.0,)
with tf.Session() as sess:
    print("加载age的关键点信息：",sess.run(a)) 
    
    #{'age': (array([17., 25., 33., 41., 49., 57., 65., 73., 81., 90.], dtype=float32), array([17.      , 25.11111 , 33.22222 , 41.333332, 49.444443, 57.555553,
#       65.666664, 73.77777 , 81.888885, 90.      ], dtype=float32))} 


#####################################################

#输出超参。用于显示
def _pprint_hparams(hparams):
  print("* hparams=[")
  for (key, value) in sorted(six.iteritems(hparams.values())):
    print("\t{}={}".format(key, value))
  print("]")

#创建calibrated_linear模型
def create_calibrated_linear(feature_columns, config, quantiles_dir):
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedLinearHParams(feature_names=feature_names, 
                            num_keypoints=200, learning_rate=1e-4)
  #hparams.parse(hparams)
  #对部分列中的超参单独赋值.
  hparams.set_feature_param("capital_gain", "calibration_l2_laplacian_reg",
                            4.0e-3)
  _pprint_hparams(hparams)
  
  return tfl.calibrated_linear_classifier(feature_columns=feature_columns,
      model_dir=config.model_dir,config=config,hparams=hparams,
      quantiles_dir=quantiles_dir)

#创建calibrated_lattice模型
def create_calibrated_lattice(feature_columns, config, quantiles_dir):
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedLatticeHParams(feature_names=feature_names,
      num_keypoints=200,lattice_l2_laplacian_reg=5.0e-3,
      lattice_l2_torsion_reg=1.0e-4,learning_rate=0.1,
      lattice_size=2)
  
  #hparams.parse(hparams)
  _pprint_hparams(hparams)
  
  return tfl.calibrated_lattice_classifier(feature_columns=feature_columns,
      model_dir=config.model_dir,config=config,
      hparams=hparams, quantiles_dir=quantiles_dir)

#创建calibrated_rtl模型
def create_calibrated_rtl(feature_columns, config, quantiles_dir):
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedRtlHParams(feature_names=feature_names,
      num_keypoints=200,learning_rate=0.02,
      lattice_l2_laplacian_reg=5.0e-4,lattice_l2_torsion_reg=1.0e-4,
      lattice_size=3,lattice_rank=4, num_lattices=100)
  #对部分列中的超参单独赋值.
  hparams.set_feature_param("capital_gain", "lattice_size", 8)
  hparams.set_feature_param("native_area", "lattice_size", 8)
  hparams.set_feature_param("marital_status", "lattice_size", 4)
  hparams.set_feature_param("age", "lattice_size", 8)
  #hparams.parse(hparams)
  _pprint_hparams(hparams)
  return tfl.calibrated_rtl_classifier(feature_columns=feature_columns,
      model_dir=config.model_dir,config=config,hparams=hparams,
      quantiles_dir=quantiles_dir)

#创建calibrated_etl模型
def create_calibrated_etl(feature_columns, config, quantiles_dir):
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedEtlHParams(feature_names=feature_names,
      num_keypoints=200,learning_rate=0.02,
      non_monotonic_num_lattices=200,non_monotonic_lattice_rank=2,
      non_monotonic_lattice_size=2,calibration_l2_laplacian_reg=4.0e-3,
      lattice_l2_laplacian_reg=1.0e-5,lattice_l2_torsion_reg=4.0e-4)
  
  #hparams.parse(hparams)
  _pprint_hparams(hparams)
  
  return tfl.calibrated_etl_classifier(feature_columns=feature_columns,
      model_dir=config.model_dir,config=config, hparams=hparams,
      quantiles_dir=quantiles_dir)


#用指定数据测试模型
def evaluate_on_data(estimator, data):
  name = os.path.basename(data) #或取输入数据的文件夹名称
  
  #评估模型
  evaluation = estimator.evaluate(input_fn=get_input_fn(  #定义输入函数
     file_path=data, batch_size=batch_size,num_epochs=1,shuffle=False),
                                  name=name)
  print("  Evaluation on '{}':\t准确率={:.4f}\t平均loss={:.4f}".format(
      name, evaluation["accuracy"], evaluation["average_loss"]))

def evaluate(estimator):#分别用训练和测试数据集，对模型进行测试 
  evaluate_on_data(estimator, traindir)
  evaluate_on_data(estimator, testdir)

def train(estimator,train_epochs,showtest = None):
  if showtest==None:  #不显示中间测试信息
    input_fn =get_input_fn(traindir, batch_size, num_epochs=train_epochs, shuffle=True)
    estimator.train(input_fn=input_fn)
  else:#训练过程中，显示10次中间模型的测试信息.
    epochs_trained = 0
    loops = 0
    while epochs_trained < train_epochs:
      loops += 1
      next_epochs_trained = int(loops * train_epochs / 10.0)
      epochs = max(1, next_epochs_trained - epochs_trained)
      epochs_trained += epochs
      input_fn =get_input_fn(traindir, batch_size, num_epochs=epochs, shuffle=True)
      estimator.train(input_fn=input_fn)
      print("Trained for {} epochs, total so far {}:".format(
          epochs, epochs_trained))
      evaluate(estimator)



allfeature_columns = create_feature_columns() #创建特征列

modelsfun = [
         create_calibrated_linear,#创建calibrated_linear模型函数
         create_calibrated_lattice,#创建calibrated_lattice模型函数
         create_calibrated_rtl,#创建calibrated_rtl模型函数
         create_calibrated_etl,#创建calibrated_etl模型函数
            ]

for modelfun in modelsfun: #依次创建函数，对其评估
    print('{0:-^50}'.format(modelfun.__name__))#分隔符
    
    output_dir = "./model_" + modelfun.__name__
    os.makedirs(output_dir, exist_ok=True)#创建模型路径
    #创建估算器配置文件
    config = tf.estimator.RunConfig().replace(model_dir=output_dir)
    
    estimator = modelfun(allfeature_columns,config, quantiles_dir)#创建估算器
    train(estimator,train_epochs=10)#训练模型,迭代10次
    evaluate(estimator)#评估模型

  