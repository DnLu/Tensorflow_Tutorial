import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist=input_data.read_data_sets("Mnist",one_hot=True)

# 每个批次大小
batch_size=100
# 计算共有多少个批次
m_batch=mnist.train.num_examples//batch_size

# 定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
# 过拟合的参数设置
keep_prob=tf.placeholder(tf.float32)
# 定义学习率
lr=tf.Variable(0.001,dtype=tf.float32)

# 创建一个稍复杂的神经网络：故意过拟合
W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1=tf.Variable(tf.zeros([500]))
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_Dropout=tf.nn.dropout(L1,keep_prob)

W2=tf.Variable(tf.truncated_normal([500,200],stddev=0.1))
b2=tf.Variable(tf.zeros([200]))
L2=tf.nn.tanh(tf.matmul(L1_Dropout,W2)+b2)
L2_Dropout=tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([200,10],stddev=0.1))
b3=tf.Variable(tf.zeros([10]))
pre=tf.nn.softmax(tf.matmul(L2_Dropout,W3)+b3)

# 定义交叉熵代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pre))

# 定义Adam的优化器，收敛速度快
train=tf.train.AdamOptimizer(lr).minimize(loss)

# 结果存放在一个布尔型变量中
# tf.arg_max返回一维张量中最大值所在的位置
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(pre,1))

# 求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 定义一个会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
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
'''Iter0,learn_rating0.001,Testing Accurary0.9038
Iter1,learn_rating0.00095,Testing Accurary0.921
Iter2,learn_rating0.0009025,Testing Accurary0.9256
Iter3,learn_rating0.000857375,Testing Accurary0.9293
Iter4,learn_rating0.000814506,Testing Accurary0.9365
Iter5,learn_rating0.000773781,Testing Accurary0.9365
Iter6,learn_rating0.000735092,Testing Accurary0.9426
Iter7,learn_rating0.000698337,Testing Accurary0.9446
Iter8,learn_rating0.00066342,Testing Accurary0.9496
Iter9,learn_rating0.000630249,Testing Accurary0.9465
Iter10,learn_rating0.000598737,Testing Accurary0.9489
Iter11,learn_rating0.0005688,Testing Accurary0.9491
Iter12,learn_rating0.00054036,Testing Accurary0.9483
Iter13,learn_rating0.000513342,Testing Accurary0.9502
Iter14,learn_rating0.000487675,Testing Accurary0.9523
Iter15,learn_rating0.000463291,Testing Accurary0.9536
Iter16,learn_rating0.000440127,Testing Accurary0.9513
Iter17,learn_rating0.00041812,Testing Accurary0.9547
Iter18,learn_rating0.000397214,Testing Accurary0.9545
Iter19,learn_rating0.000377354,Testing Accurary0.9555'''
