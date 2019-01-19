# -*- coding: utf-8 -*-

from skimage import io, transform
import glob
import numpy as np
import time
import CNN
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf


path = '/home/zmx/FaceRecognition_using_Tensorflow/datasets/face'

w = 128
h = 128
c = 3


# 读取图片
def read_img(path):
    n = 0   # 计数
    # 照片路径
    cate = [path + '/' + x for x in sorted(os.listdir(path)) if os.path.isdir(path + '/' + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in sorted(glob.glob(folder + '/*.png')):
            print('reading the images:%s' % (im))
            print(idx)
            print(folder)
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
            n += 1
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)

# 打乱顺序
num_example = data.shape[0]     # data大小
arr = np.arange(num_example)
np.random.shuffle(arr)  # 打乱
data = data[arr]
label = label[arr]

ratio = 0.8
s = np.int(num_example*ratio)
# 前80%为训练集
x_train = data[:s]
y_train = label[:s]
# 剩余为测试集
x_val = data[s:]
y_val = label[s:]


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


logits = CNN.CNNlayer(x)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练和测试数据
saver = tf.train.Saver(max_to_keep=3)
max_acc = 0
f = open('train/acc.txt', 'w')

n_epoch = 10
batch_size = 64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))

    f.write(str(epoch + 1) + ', val_acc: ' + str(val_acc) + '\n')
    if val_acc > max_acc:
        max_acc = val_acc
        saver.save(sess, 'train/faces.ckpt', global_step=epoch + 1)

f.close()
sess.close()
