import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]#[:,np.newaxis]使其变成矩阵
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

# 定义2个placeholder
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

# 定义神经网络的中间层
W_L1=tf.Variable(tf.random_normal([1,10]))
b_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,W_L1)+b_L1
L1=tf.nn.relu(Wx_plus_b_L1)
# 定义神经网络的输出层
W_L2=tf.Variable(tf.random_normal([10,1]))
b_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,W_L2)+b_L2
Pre=tf.nn.tanh(Wx_plus_b_L2)

# 定义二次代价损失函数
loss=tf.reduce_mean(tf.square(y-Pre))

# 定义一个梯度下降法的优化器
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化参数
init=tf.global_variables_initializer()

# 定义一个会话
with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # 获得预测值
    Prediction=sess.run(Pre,feed_dict={x:x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,Prediction,'r-',lw=5)
    plt.show()