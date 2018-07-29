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

# 创建一个简单的神经网络
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
pre=tf.nn.softmax(tf.matmul(x,W)+b)

# 定义二次代价函数
loss=tf.reduce_mean(tf.square(y-pre))

# 定义梯度下降法的优化器
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 结果存放在一个布尔型变量中
# tf.arg_max返回一维张量中最大值所在的位置
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(pre,1))

# 求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 定义一个会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(m_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        # 计算准确率
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accurary:"+str(acc))
'''Iter0,Testing Accurary0.7417
Iter1,Testing Accurary0.833
Iter2,Testing Accurary0.8594
Iter3,Testing Accurary0.8705
Iter4,Testing Accurary0.8778
Iter5,Testing Accurary0.8822
Iter6,Testing Accurary0.8852
Iter7,Testing Accurary0.8878
Iter8,Testing Accurary0.8912
Iter9,Testing Accurary0.8941
Iter10,Testing Accurary0.896
Iter11,Testing Accurary0.8969
Iter12,Testing Accurary0.8985
Iter13,Testing Accurary0.8992
Iter14,Testing Accurary0.9008
Iter15,Testing Accurary0.9018
Iter16,Testing Accurary0.9022
Iter17,Testing Accurary0.9034
Iter18,Testing Accurary0.9045
Iter19,Testing Accurary0.9049
Iter20,Testing Accurary0.9059'''
