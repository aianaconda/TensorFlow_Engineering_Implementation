"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
from datetime import datetime
import math
import os
import time
import tensorflow as tf
import traceback
import numpy as np
from scipy import signal
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt

tacotron = __import__("9-16  tacotron")
Tacotron = tacotron.Tacotron

cn_dataset = __import__("9-15  cn_dataset")
mydataset = cn_dataset.mydataset
sequence_to_text = cn_dataset.sequence_to_text

def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')

max_frame_num=1000  #定义每个音频文件的最大帧数
sample_rate=16000  #定义音频文件的采样率
num_mels=80
num_freq=1025
outputs_per_step = 5

n_fft = (num_freq - 1) * 2   #stft算法中使用的窗口大小（因为声音的真实频率只有正的，而fft变化是对称的，需要加上负频率）
frame_length_ms=50 #定义stft算法中的重叠窗口（用时间来表示）
frame_shift_ms=12.5#定义stft算法中的移动步长（用时间来表示）
hop_length = int(frame_shift_ms / 1000 * sample_rate)#定义stft算法中的帧移步长
win_length = int(frame_length_ms / 1000 * sample_rate)#定义stft算法中的相邻两个窗口的重叠长度


preemphasis=0.97#用于过滤声音频率的阀值
ref_level_db=20 #控制峰值得阀值
min_level_db=-100#指定dB最小值，用于归一化


griffin_lim_iters=60    #Griffin-Lim算法合成语音时的计算次数
power=1.5               #设置在Griffin-Lim算法之前，提升振幅的参数

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def inv_preemphasis(x):#使用数字滤波器进行音频信号恢复
  return signal.lfilter([1], [1, -preemphasis], x)


def _griffin_lim(S):#使用griffin lim信号估计算法，恢复声音
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  #反向短时傅里叶变换
  y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
  for i in range(griffin_lim_iters):
    angles = np.exp(1j * np.angle(librosa.stft(y,n_fft,hop_length,win_length)))
    y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
  return y

def inv_spectrogram(spectrogram):  #将特征信号转换成wave形式的声音
  S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  #将dB频谱转为音频特征信号
  return inv_preemphasis(_griffin_lim(S ** power))          

def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  #librosa.output.write_wav(path, wav.astype(np.int16), sample_rate)
  wavfile.write(path, sample_rate, wav.astype(np.int16))

def train(log_dir):#训练模型

  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  tf.logging.info('Checkpoint path: %s' % checkpoint_path)
  #加载数据集
  next_element = mydataset('training/train.txt',outputs_per_step=5)

  #定义输入占位符
  inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
  input_lengths= tf.placeholder(tf.int32, [None], 'input_lengths')
  linear_targets =     tf.placeholder(tf.float32, [None, None, num_freq], 'linear_targets')
  mel_targets =    tf.placeholder(tf.float32, [None, None, num_mels], 'mel_targets')
  stop_token_targets =     tf.placeholder(tf.float32, [None, None], 'stop_token_targets')
  
  #构建网络模型
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = Tacotron(inputs, input_lengths,num_mels,outputs_per_step,num_freq,
                     linear_targets, mel_targets,  stop_token_targets)
    model.buildTrainModel(sample_rate,num_freq,global_step)



  time_window = []
  loss_window = []
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  eporch = 100000 #定义迭代训练次数
  checkpoint_interval = 1000#每1000次，保存一次检查点
  
   
  os.makedirs(log_dir, exist_ok=True)
  checkpoint_state = tf.train.get_checkpoint_state(log_dir)
  
  def plot_alignment(alignment, path, info=None):#输出音频图谱
      fig, ax = plt.subplots()
      im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
      fig.colorbar(im, ax=ax)
      xlabel = 'Decoder timestep'
      if info is not None:
        xlabel += '\n\n' + info
      plt.xlabel(xlabel)
      plt.ylabel('Encoder timestep')
      plt.tight_layout()
      plt.savefig(path, format='png')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #恢复检查点
    if checkpoint_state is not None:
        saver.restore(sess, checkpoint_state.model_checkpoint_path)
        tf.logging.info('Resuming from checkpoint: %s ' % (checkpoint_state.model_checkpoint_path) )
    else:
        tf.logging.info('Starting new training ')

    try:#迭代训练
        for i in range(eporch):
            seq,seqlen,linear_target,mel_target,stop_token_target = sess.run(next_element)

            start_time = time.time()
            step, loss, opt = sess.run([global_step, model.loss, model.optimize], 
                                       feed_dict={inputs: seq, input_lengths: seqlen, 
                                             linear_targets: linear_target,mel_targets: mel_target,  
                                             stop_token_targets: stop_token_target})
    
            time_window.append(time.time() - start_time)
            loss_window.append(loss)
            message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
              step, sum(time_window) / max(1, len(time_window)),
              loss,  sum(loss_window) / max(1, len(loss_window)))
            tf.logging.info(message)

            if loss > 100 or math.isnan(loss):
              tf.logging.info('Loss exploded to %.05f at step %d!' % (loss, step))
              raise Exception('Loss Exploded')
    
            if step % checkpoint_interval == 0:
              tf.logging.info('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
              saver.save(sess, checkpoint_path, global_step=step)
              tf.logging.info('Saving audio and alignment...')
              #输出模型结果
              input_seq, spectrogram, alignment = sess.run([
                model.inputs[0], model.linear_outputs[0], model.alignments[0]], 
                                       feed_dict={inputs: seq, input_lengths: seqlen, 
                                             linear_targets: linear_target,mel_targets: mel_target,  
                                             stop_token_targets: stop_token_target})
    
              waveform = inv_spectrogram(spectrogram.T)#转成音频数据
              #保存音频数据
              save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
              #绘制音频图谱
              plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                    info=' %s, step=%d, loss=%.5f' % (  time_string(), step, loss))
              tf.logging.info('Input: %s' % sequence_to_text(input_seq))

    except Exception as e:
        tf.logging.info('Exiting due to exception: %s' % e)
        traceback.print_exc()


if __name__ == '__main__':
  tf.reset_default_graph()#重置图
  tf.logging.set_verbosity(tf.logging.INFO)#定义输出的log级别
  train(os.path.join('.', 'model-cpk' )) #训练模型