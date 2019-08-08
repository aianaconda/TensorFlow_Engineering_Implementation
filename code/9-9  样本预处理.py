"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor #载入多进程库
from functools import partial
import numpy as np
import glob
from scipy import signal
import librosa

max_frame_num=1000  #定义每个音频文件的最大帧数
sample_rate=16000  #定义音频文件的采样率

num_freq=1025 #振幅频率 
num_mels=80 #定义Mel bands特征个数

frame_length_ms=50 #定义stft算法中的重叠窗口（用时间来表示）
frame_shift_ms=12.5#定义stft算法中的移动步长（用时间来表示）

preemphasis=0.97#用于数字滤波器的阀值

n_fft = (num_freq - 1) * 2   #stft算法中使用的窗口大小（因为声音的真实频率只有正的，而fft变化是对称的，需要加上负频率）
hop_length = int(frame_shift_ms / 1000 * sample_rate)#定义stft算法中的帧移步长
win_length = int(frame_length_ms / 1000 * sample_rate)#定义stft算法中的相邻两个窗口的重叠长度

ref_level_db=20 #控制峰值得阀值
min_level_db=-100#指定dB最小值，用于归一化

#创建一个Mel filter。， shape=(n_mels, 1 + n_fft/2)即(n_mels, num_freq)
_mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

def spectrogram(D):#转为db频谱
  S =20 * np.log10(np.maximum(1e-5, D)) - ref_level_db  #将幅度频谱转为db频谱
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)#归一化

def melspectrogram(D):#转化为mel特征
  mel = np.dot(_mel_basis, D)#通过与Filterbank矩阵点积计算，将分帧结果转化为mel特征
  return spectrogram(mel)

def _process_utterance(out_dir, index, wav_path,pinyin):#进程处理函数

  #按照16000的采样率 读取音频
  wav,_ = librosa.core.load(wav_path, sr=sample_rate) 
 
  #对波形文件，进行数字滤波处理
  emphasis = signal.lfilter([1, -preemphasis], [1], wav)

  #短时傅里叶变换 进行音频分帧
  D=np.abs(librosa.stft(emphasis, n_fft, hop_length, win_length))

  #计算原始声音分帧后的时频图
  linear_spectrogram = spectrogram(D).astype(np.float32)
  n_frames = linear_spectrogram.shape[1] #返回帧的个数
  if n_frames > max_frame_num:
    return None

  #计算原始声音分帧后的mel特征时频图
  mel_spectrogram = melspectrogram(D).astype(np.float32)

  #保存转化后的特征数据
  spectrogram_filename = 'thchs30-spec-%05d.npy' % index
  mel_filename = 'thchs30-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), linear_spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  #返回特征文件名即样本的拼音标注
  return (spectrogram_filename, mel_filename, n_frames,pinyin)


#多进程实现音频样本数据转化
def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):

    executor = ProcessPoolExecutor(max_workers=num_workers)#创建进程池执行器
    futures = []
    index = 1
    #获取指定目录下的文件
    wav_files = glob.glob(os.path.join(in_dir, 'mytest', '*.wav'))

    #读取标注文件
    with open(os.path.join(in_dir, r'doc/trans', 'test.syllable.txt')) as f:
        allpinyin = {}
        for pinyin in f:
            indexf = pinyin.index(' ')
            allpinyin[pinyin[:indexf]] = pinyin[indexf+1:]
          
    #将音频文件与标注关联一起
    for wav_file in wav_files:
        key = wav_file[ wav_file.index('D'):-4]
        task = partial(_process_utterance, out_dir, index, wav_file,allpinyin[key])
        futures.append(executor.submit(task))
        index += 1
    return [future.result() for future in tqdm(futures) if future.result() is not None]

    

def preprocess_data(num_workers):
    in_dir = os.path.join(os.path.expanduser('.'), 'data_thchs30')
    out_dir = os.path.join(os.path.expanduser('.'), 'training')
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)

    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]))
    frames = sum([m[2] for m in metadata])

    print('Wrote %d utterances, %d frames ' % (len(metadata), frames))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))
    
    
def main():
    preprocess_data(cpu_count())


if __name__ == "__main__":
  main()
