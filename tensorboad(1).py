import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist=input_data.read_data_sets("Mnist",one_hot=True)


# 每个批次大小
batch_size=100
# 计算共有多少个批次
m_batch=mnist.train.num_examples//batch_size
# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x=tf.placeholder(tf.float32,[None,784],name='x_input')
    y=tf.placeholder(tf.float32,[None,10],name='y_input')
# 过拟合的参数设置
keep_prob=tf.placeholder(tf.float32)
# 定义学习率
lr=tf.Variable(0.001,dtype=tf.float32)
# 命名空间
with tf.name_scope('network'):
    # 创建一个稍复杂的神经网络：故意过拟合
    with tf.name_scope('layer1'):
        with tf.name_scope('W1'):
            W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
        with tf.name_scope('b1'):
            b1=tf.Variable(tf.zeros([500]))
        with tf.name_scope('L1_Wx_plus_b'):
            L1_Wx_plus_b=tf.matmul(x,W1)+b1
        with tf.name_scope('L1_tanh'):
            L1=tf.nn.tanh(L1_Wx_plus_b)
        with tf.name_scope('L1_Dropout'):
            L1_Dropout=tf.nn.dropout(L1,keep_prob)
    with tf.name_scope('layer2'):
        with tf.name_scope('W2'):
            W2=tf.Variable(tf.truncated_normal([500,200],stddev=0.1))
        with tf.name_scope('b2'):
            b2=tf.Variable(tf.zeros([200]))
        with tf.name_scope('L2_Wx_plus_b'):
            L2_Wx_plus_b = tf.matmul(L1_Dropout, W2) + b2
        with tf.name_scope('L2_tanh'):
            L2=tf.nn.tanh(L2_Wx_plus_b)
        with tf.name_scope('L2_Dropout'):
            L2_Dropout=tf.nn.dropout(L2,keep_prob)
    with tf.name_scope('layer3'):
        with tf.name_scope('W3'):
            W3=tf.Variable(tf.truncated_normal([200,10],stddev=0.1))
        with tf.name_scope('b3'):
            b3=tf.Variable(tf.zeros([10]))
        with tf.name_scope('L1_Wx_plus_b'):
            L3_Wx_plus_b = tf.matmul(L2_Dropout, W3) + b3
        with tf.name_scope('L1_tanh'):
            pre=tf.nn.softmax(L3_Wx_plus_b)

# 定义交叉熵代价函数
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pre))
    tf.summary.scalar('loss',loss)

# 定义Adam的优化器，收敛速度快
with tf.name_scope('train'):
    train=tf.train.AdamOptimizer(lr).minimize(loss)

# 结果存放在一个布尔型变量中
# tf.arg_max返回一维张量中最大值所在的位置
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(pre,1))

# 求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 定义一个会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(1):
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(m_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        # 更新学习率、计算准确率
        learn_rating=sess.run(lr)
        Test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.5})
        print("Iter"+str(epoch)+",learn_rating:"+str(learn_rating)+",Testing Accurary:"+str(Test_acc))
        # Train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.5})
        # print("Iter" + str(epoch) + ",Training Accurary" + str(Train_acc))

