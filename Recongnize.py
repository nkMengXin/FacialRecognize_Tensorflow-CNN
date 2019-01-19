# coding:utf-8
from skimage import io, transform
import cv2
import dlib
import glob
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import time
import sys
import CNN
w = 128
h = 128
c = 3
detector = dlib.get_frontal_face_detector()  # 获取人脸分类器

IDpath = "/home/zmx/FaceRecognition_using_Tensorflow/datasets/temp"
dirs = os.listdir(IDpath)
ID = [dir for dir in sorted(dirs)]


x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


logits = CNN.CNNlayer(x)
predict = tf.argmax(logits, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'train/faces.ckpt-12')

user = input("图片（1）还是摄像头（2）:")
if user == "1":
    path = input("图片路径名是：")
    img = cv2.imread(path)
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for index, face in enumerate(dets):
        # 输出人脸位置
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                     face.bottom()))
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        # 用矩形标出人脸
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.imshow('image', img)
        # 裁出人脸
        facecut = img[top:bottom, left:right]
        io.imsave('temp.png', facecut)
        img1 = io.imread('temp.png')
        # 将裁出的人脸resize为需要的维度
        img1 = transform.resize(img1, (w, h, c))
        # 输入神经网络
        res = sess.run(predict, feed_dict={x: [img1]})
        print(ID[res[0]])
    if len(dets) == 0:
    # 若未检测到人脸
        img = transform.resize(img, (w, h, c))
        res = sess.run(predict, feed_dict={x: [img]})
        print(ID[res[0]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 视屏封装格式

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)

        # 抓取图像
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('image/now.png', frame)

            img = cv2.imread("image/now.png")
            dets = detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for index, face in enumerate(dets):
                print('face {}; left {}; top {}; right {}; bottom {}'.format(index,
                                                                             face.left(), face.top(), face.right(),
                                                                             face.bottom()))
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                img = img[top:bottom, left:right]

            # img=io.imread('image/now.png')
            img = transform.resize(img, (w, h, c))
            res = sess.run(predict, feed_dict={x: [img]})
            print(ID[res[0]])

        # 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
