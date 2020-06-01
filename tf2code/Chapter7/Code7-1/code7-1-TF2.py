# -*- coding: utf-8 -*-


import argparse               #引入系统头文件
import os
import shutil
import sys

import tensorflow as tf       #引TensorFlow入头文件
from utils import parsers,hooks_helper,model_helpers  #引入utils头文件
#tf.compat.v1.disable_v2_behavior()#本例中非必要

_CSV_COLUMNS = [                                #定义CVS列名
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_area',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [                        #定义每一列的默认值
        [0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {                               #定义样本集的数量
    'train': 32561,
    'validation': 16281,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}#定义模型的前缀


def build_model_columns():
  """生成wide和deep模型的特征列集合."""
  #定义连续值列
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')

  #定义离散值列，返回的是稀疏矩阵
  education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      'workclass', [
          'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

  #将所有职业名称通过hash算法，离散成1000个类别:
  occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=1000)

  #将连续值特征列转为离散值特征.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  #定义基础特征列.
  base_columns = [
      education, marital_status, relationship, workclass, occupation,
      age_buckets,
  ]
  #定义交叉特征列.
  crossed_columns = [
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
  ]

  #定义wide模型的特征列.
  wide_columns = base_columns + crossed_columns

  #定义deep模型的特征列.
  deep_columns = [
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(workclass),              #将workclass列的稀疏矩阵转成0ne_hot编码
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      tf.feature_column.embedding_column(occupation, dimension=8),#将1000个hash后的类别，每个用嵌入词embedding转换
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type,warm_start_from=None):
  """按照指定的模型生成估算器对象."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  run_config = tf.estimator.RunConfig().replace(                #将GPU个数设为0，关闭GPU运算。因为该模型在CPU上速度更快
      session_config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}),
      save_checkpoints_steps=1000)

  if model_type == 'wide':                                      #生成带有wide模型的估算器对象
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config, loss_reduction=tf.keras.losses.Reduction.SUM)
  elif model_type == 'deep':                                    #生成带有deep模型的估算器对象
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config, loss_reduction=tf.keras.losses.Reduction.SUM)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(            #生成带有wide和deep模型的估算器对象
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        warm_start_from=warm_start_from, loss_reduction=tf.keras.losses.Reduction.SUM)


def input_fn(data_file, num_epochs, shuffle, batch_size):       #定义估算器输入函数
  """估算器的输入函数."""
  assert tf.io.gfile.exists(data_file), (                          #用断言语句判断样本文件是否存在
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.io.decode_csv(records=value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    return features, tf.equal(labels, '>50K')


  dataset = tf.data.TextLineDataset(data_file)                  #创建dataset数据集

  if shuffle:                                                   #对数据进行乱序操作
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)        #对data_file文件中的每行数据，进行特征抽取，返回新的数据集

  dataset = dataset.repeat(num_epochs)                          #将数据集重复num_epochs次
  dataset = dataset.batch(batch_size)                           #将数据集按照batch_size划分
  dataset = dataset.prefetch(1)
  return dataset


def export_model(model, model_type, export_dir):                #定义函数export_model ，用于导出模型
  """导出模型.

  参数:
    model: 估算器对象
    model_type: 要导出的模型类型，可选值有 "wide"、 "deep" 或 "wide_deep"
    export_dir: 导出模型的路径.
  """
  wide_columns, deep_columns = build_model_columns()        #获得列张量
  if model_type == 'wide':
    columns = wide_columns
  elif model_type == 'deep':
    columns = deep_columns
  else:
    columns = wide_columns + deep_columns
  feature_spec = tf.feature_column.make_parse_example_spec(columns)
  example_input_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
  model.export_saved_model(export_dir, example_input_fn)


class WideDeepArgParser(argparse.ArgumentParser):               #定义WideDeepArgParser类，用于解析参数
  """该类用于在程序启动时的参数解析."""

  def __init__(self):                                           #初始化函数
    super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()]) #调用父类的初始化函数
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide_deep',   #添加一个启动参数--model_type，默认值为wide_deep
        choices=['wide', 'deep', 'wide_deep'],                  #定义该参数的可选值
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.', #定义启动参数的帮助命令
        metavar='<MT>')
    self.set_defaults(                                          #为其他参数设置默认值
        data_dir='income_data',                                 #设置数据样本路径
        model_dir='income_model',                               #设置模型存放路径
        export_dir='income_model_exp',                          #设置导出模型存放路径
        train_epochs=5,                                        #设置迭代次数
        batch_size=40)                                          #设置批次大小

def trainmain(argv):
  parser = WideDeepArgParser()                                  #实例化WideDeepArgParser，用于解析启动参数
  flags = parser.parse_args(args=argv[1:])                      #获得解析后的参数flags
  print("解析的参数为：",flags)

  shutil.rmtree(flags.model_dir, ignore_errors=True)            #如果模型存在，整个目录删除
  model = build_estimator(flags.model_dir, flags.model_type)    #生成估算器对象

  train_file = os.path.join(flags.data_dir, 'adult.data.csv')       #获得训练集样本文件的路径
  test_file = os.path.join(flags.data_dir, 'adult.test.csv')        #获得测试集样本文件的路径


  def train_input_fn():                                         #定义训练集样本输入函数
    return input_fn(                                            #该输入函数按照batch_size批次,迭代输入epochs_between_evals次，使用乱序处理
        train_file, flags.epochs_between_evals, True, flags.batch_size)

  def eval_input_fn():                                          #定义测试集样本输入函数
    return input_fn(test_file, 1, False, flags.batch_size)      #该输入函数按照batch_size批次,迭代输入1次，不使用乱序处理

  loss_prefix = LOSS_PREFIX.get(flags.model_type, '')           #格式化输出loss的前缀

  for n in range(flags.train_epochs ): #将总迭代数，按照epochs_between_evals分段。并循环对每段进行训练#调用估算器的train方法进行训练  
    model.train(input_fn=train_input_fn)         #调用估算器的train方法进行训练
    results = model.evaluate(input_fn=eval_input_fn)                #调用估算器的evaluate方法进行评估计算

    print('{0:-^60}'.format('evaluate at epoch %d'%( (n + 1))))#分隔符

    for key in sorted(results):                                     #显示评估结果
      print('%s: %s' % (key, results[key]))

    if model_helpers.past_stop_threshold(                           #根据accuracy的阈值，来判断是否需要结束训练。
        flags.stop_threshold, results['accuracy']):
      break

  if flags.export_dir is not None:                                  #根据设置导出冻结图模型，用于tfseving
    export_model(model, flags.model_type, flags.export_dir)




def premain(argv):
  parser = WideDeepArgParser()                                  #实例化WideDeepArgParser，用于解析启动参数
  flags = parser.parse_args(args=argv[1:])                      #获得解析后的参数flags
  print("解析的参数为：",flags)

  test_file = os.path.join(flags.data_dir, 'adult.test.csv')        #获得测试集样本文件的路径

  def eval_input_fn():                                          #定义测试集样本输入函数
    return input_fn(test_file, 1, False, flags.batch_size)      #该输入函数按照batch_size批次,迭代输入1次，不使用乱序处理

    #model2 = build_estimator('./temp', flags.model_type,flags.model_dir)#也可以使用热启动的方式
  model2 = build_estimator(flags.model_dir, flags.model_type)

  predictions = model2.predict(input_fn=eval_input_fn)
  for i, per in enumerate(predictions):
      print("csv中第",i,"条结果为：",per['class_ids'])
      if i==5:
          break

if __name__ == '__main__':                  #当运行文件时，模块名字__name__就会为__main__
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #设置log级别为INFO，如果想要显示的信息少点，可以设置成 WARN
  trainmain(argv=sys.argv)                       #调用main函数，进入程序主体
  premain(argv=sys.argv)
