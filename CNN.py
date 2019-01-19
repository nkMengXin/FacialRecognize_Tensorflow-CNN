import tensorflow as tf
w = 128
h = 128
c = 3


# -----------------构建网络----------------------
def CNNlayer(x):    # 参数x为输入的占位符
    # 第一个卷积层（128——>64)
    conv1 = tf.layers.conv2d(
        inputs=x,                   # inputs：Tensor 输入
        filters=32,                 # filters：整数，表示输出空间的维数（即卷积过滤器的数量）
        kernel_size=[5, 5],         # kernel_size：卷积核的大小
        padding="same",             # padding："valid" 或者 "same"
                                    # （"valid" 表示不够卷积核大小的块就丢弃，"same"表示不够卷积核大小的块就补0。）
        activation=tf.nn.relu,      # activation：激活函数
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))    # kernel_initializer：卷积核的初始化。
    # 第一个池化层
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层(64->32)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层(32->16)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层(16->8)
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense2,
                             units=69,      # 测试时此处为13（选取了13个人的数据）
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits
