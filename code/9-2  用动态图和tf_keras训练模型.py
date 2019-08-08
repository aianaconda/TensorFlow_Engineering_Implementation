# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import json
from PIL import Image
import tensorflow as tf

import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

annotation_file = r'cocos2014/annotations/captions_train2014.json'
PATH = r"cocos2014/train2014/"
numpyPATH = './numpyfeature/'

preimgdata = __import__("9-3  利用Resnet进行样本预处理")
makenumpyfeature = preimgdata.makenumpyfeature

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

#加载指定个数的图片路径和对应的标题
num_examples = 300
train_captions = []
img_filename= []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path =  'COCO_train2014_' + '%012d.jpg' % (image_id)

    img_filename.append(full_coco_image_path)
    train_captions.append(caption)
    if len(train_captions) >=num_examples:
        break

print(img_filename[:3])

if not os.path.exists(numpyPATH):
    makenumpyfeature(numpyPATH,img_filename,PATH)

#############################################
#过滤文本，选出5000个
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
#train_seqs = tokenizer.texts_to_sequences(train_captions)
#print(train_seqs[0])

#构造字典
tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value <= top_k}
print(tokenizer.word_index)
# putting <unk> token in the word2idx dictionary
tokenizer.word_index[tokenizer.oov_token] = top_k + 1
tokenizer.word_index['<pad>'] = 0
print(tokenizer.word_index)

#反向字典
index_word = {value:key for key, value in tokenizer.word_index.items()}
print(index_word)

#变为向量
train_seqs = tokenizer.texts_to_sequences(train_captions)
print(train_seqs[0])

#按照最长的句子对齐。不足的在其后面补0
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
print("最大长度",len(cap_vector[0]))
max_length =len(cap_vector[0])

#将数据拆成训练和测试
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_filename,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)


BATCH_SIZE = 20
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index)

#图片特征(47, 2048)
features_shape = 2048
attention_features_shape = 49

#加载numpy 文件
def map_func(img_name, cap):
    img_tensor = np.load(numpyPATH+img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

#用map加载numpy特征文件
dataset = dataset.map(lambda item1, item2: tf.py_function(
          map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)

dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(1)

##########
#构建模型
class DNN_Encoder(tf.keras.Model):#编码器模型
    def __init__(self, embedding_dim):
        super(DNN_Encoder, self).__init__()
        #keras的全连接支持多维输入。仅对最后一维进行处理
        self.fc = tf.keras.layers.Dense(embedding_dim)#(batch_size, 49, embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

#注意力模型
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, #features形状(batch_size, 49, embedding_dim)
           hidden):#hidden(batch_size, hidden_size)

    hidden_with_time_axis = tf.expand_dims(hidden, 1)#(batch_size, 1, hidden_size)

    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))#(batch_size, 49, hidden_size)

    attention_weights = tf.nn.softmax(self.V(score), axis=1)#(batch_size, 49, 1)

    context_vector = attention_weights * features#(batch_size, 49, hidden_size)
    context_vector = tf.reduce_sum(context_vector, axis=1)#(batch_size,  hidden_size)

    return context_vector, attention_weights

def gru(units):
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = gru(self.units)
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    #返回注意力特征向量和注意力权重
    context_vector, attention_weights = self.attention(features, hidden)

    x = self.embedding(x)#(batch_size, 1, embedding_dim)

    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)#(batch_size, 1, embedding_dim + hidden_size)

    output, state = self.gru(x)#使用循环网络gru进行处理

    x = self.fc1(output)#(batch_size, max_length, hidden_size)

    x = tf.reshape(x, (-1, x.shape[2]))#(batch_size * max_length, hidden_size)

    x = self.fc2(x)#(batch_size * max_length, vocab)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))



# We are masking the loss calculated for padding
def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)#批次中被补0的序列不参与计算loss
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

def all_loss(encoder,decoder,img_tensor,target):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
    features = encoder(img_tensor)#(20, 49, 256)


    for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        loss += loss_function(target[:, i], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)
    return loss


grad = tfe.implicit_gradients(all_loss)

#创建模型对象字典
model_objects = {
        'encoder':DNN_Encoder(embedding_dim),
        'decoder' :RNN_Decoder(embedding_dim, units, vocab_size) ,
        'optimizer': tf.train.AdamOptimizer(),
        'step_counter': tf.train.get_or_create_global_step(),
}

checkpoint_prefix = os.path.join("mytfemodel/", 'ckpt')
checkpoint = tf.train.Checkpoint(**model_objects)
latest_cpkt = tf.train.latest_checkpoint("mytfemodel/")
if latest_cpkt:
    print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint.restore(latest_cpkt)

#实现单步训练过程
def train_one_epoch(encoder,decoder,optimizer,step_counter,dataset,epoch):
    total_loss = 0
    for (step, (img_tensor, target)) in enumerate(dataset):
        loss = 0

        optimizer.apply_gradients(grad(encoder,decoder,img_tensor, target),step_counter)
        loss =all_loss(encoder,decoder,img_tensor, target)

        total_loss += (loss / int(target.shape[1]))
        if step % 5 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                          step,
                                                         loss.numpy() / int(target.shape[1])))
    print("step",step)
    return total_loss/(step+1)

#训练模型
loss_plot = []
EPOCHS = 50

for epoch in range(EPOCHS):
    start = time.time()
    total_loss= train_one_epoch(dataset=dataset,epoch=epoch,**model_objects)#训练一次

    loss_plot.append(total_loss )#保存loss

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss))
    checkpoint.save(checkpoint_prefix)
    print('Train time for epoch #%d (step %d): %f' %
    (checkpoint.save_counter.numpy(),  checkpoint.step_counter.numpy(), time.time() - start))

#
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()

def evaluate(encoder,decoder,optimizer,step_counter,image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)
    size = [224,224]
    def load_image(image_path):
        img = tf.read_file(PATH +image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, image_path
    from tensorflow.python.keras.applications.resnet50 import ResNet50

    image_model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
                     ,include_top=False)#创建ResNet网络

    new_input = image_model.input
    hidden_layer = image_model.layers[-2].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

#    print(step_counter.numpy())
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        print(predictions.get_shape())

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
        result.append(index_word[predicted_id])

        print(predicted_id)

        if index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(PATH +image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        #print(len(attention_plot[l]),attention_plot[l])
        temp_att = np.resize(attention_plot[l], (7, 7))
        ax = fig.add_subplot(len_result//2, len_result//2+len_result%2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.4, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

# captions on the validation set
rid = np.random.randint(0, len(img_name_val))


image = img_name_val[rid]
real_caption = ' '.join([index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image=image,**model_objects)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)
## opening the image
img = Image.open(PATH +img_name_val[rid])
plt.imshow(img)
plt.axis('off')
plt.show()


