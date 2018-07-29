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

# 创建一个稍复杂的神经网络：故意过拟合
W1=tf.Variable(tf.truncated_normal([784,1000],stddev=0.1))
b1=tf.Variable(tf.zeros([1000]))
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_Dropout=tf.nn.dropout(L1,keep_prob)

W2=tf.Variable(tf.truncated_normal([1000,200],stddev=0.1))
b2=tf.Variable(tf.zeros([200]))
L2=tf.nn.tanh(tf.matmul(L1_Dropout,W2)+b2)
L2_Dropout=tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([200,100],stddev=0.1))
b3=tf.Variable(tf.zeros([100]))
L3=tf.nn.tanh(tf.matmul(L2_Dropout,W3)+b3)
L3_Dropout=tf.nn.dropout(L3,keep_prob)

W4=tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10]))
pre=tf.nn.softmax(tf.matmul(L3_Dropout,W4)+b4)

# 定义交叉熵代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pre))

# 定义梯度下降法的优化器
train=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 结果存放在一个布尔型变量中
# tf.arg_max返回一维张量中最大值所在的位置
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(pre,1))

# 求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 定义一个会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(200):
        for batch in range(m_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        # 计算准确率
        Test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.5})
        Train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.5})
        print("Iter"+str(epoch)+",Testing Accurary:"+str(Test_acc)+ ",Training Accurary:" + str(Train_acc))

'''Iter0,Testing Accurary0.7947,Training Accurary0.790527
Iter1,Testing Accurary0.8578,Training Accurary0.849018
Iter2,Testing Accurary0.874,Training Accurary0.870727
Iter3,Testing Accurary0.8878,Training Accurary0.879818
Iter4,Testing Accurary0.8921,Training Accurary0.887091
Iter5,Testing Accurary0.8966,Training Accurary0.892127
Iter6,Testing Accurary0.9006,Training Accurary0.897255
Iter7,Testing Accurary0.9036,Training Accurary0.902291
Iter8,Testing Accurary0.9066,Training Accurary0.905764
Iter9,Testing Accurary0.9077,Training Accurary0.907818
Iter10,Testing Accurary0.9092,Training Accurary0.909109
Iter11,Testing Accurary0.9124,Training Accurary0.912418
Iter12,Testing Accurary0.9124,Training Accurary0.9152
Iter13,Testing Accurary0.9177,Training Accurary0.916855
Iter14,Testing Accurary0.9183,Training Accurary0.919036
Iter15,Testing Accurary0.9243,Training Accurary0.920836
Iter16,Testing Accurary0.918,Training Accurary0.920145
Iter17,Testing Accurary0.9231,Training Accurary0.922455
Iter18,Testing Accurary0.9225,Training Accurary0.9242
Iter19,Testing Accurary0.9227,Training Accurary0.925655
Iter20,Testing Accurary0.9221,Training Accurary0.925418
Iter21,Testing Accurary0.9256,Training Accurary0.926
Iter22,Testing Accurary0.9253,Training Accurary0.929018
Iter23,Testing Accurary0.9286,Training Accurary0.929818
Iter24,Testing Accurary0.9282,Training Accurary0.9292
Iter25,Testing Accurary0.9287,Training Accurary0.931709
Iter26,Testing Accurary0.9297,Training Accurary0.930145
Iter27,Testing Accurary0.9315,Training Accurary0.932945
Iter28,Testing Accurary0.9317,Training Accurary0.932673
Iter29,Testing Accurary0.9336,Training Accurary0.934982
Iter30,Testing Accurary0.9296,Training Accurary0.934273
Iter31,Testing Accurary0.9325,Training Accurary0.934564
Iter32,Testing Accurary0.9327,Training Accurary0.935382
Iter33,Testing Accurary0.9346,Training Accurary0.937127
Iter34,Testing Accurary0.9342,Training Accurary0.9376
Iter35,Testing Accurary0.9308,Training Accurary0.938109
Iter36,Testing Accurary0.9385,Training Accurary0.937273
Iter37,Testing Accurary0.9374,Training Accurary0.939309
Iter38,Testing Accurary0.9337,Training Accurary0.939091
Iter39,Testing Accurary0.9378,Training Accurary0.939909
Iter40,Testing Accurary0.9367,Training Accurary0.9412
Iter41,Testing Accurary0.936,Training Accurary0.941927
Iter42,Testing Accurary0.9403,Training Accurary0.941745
Iter43,Testing Accurary0.9362,Training Accurary0.941036
Iter44,Testing Accurary0.9414,Training Accurary0.942927
Iter45,Testing Accurary0.9379,Training Accurary0.942509
Iter46,Testing Accurary0.9401,Training Accurary0.943491
Iter47,Testing Accurary0.9382,Training Accurary0.943891
Iter48,Testing Accurary0.9408,Training Accurary0.943073
Iter49,Testing Accurary0.9413,Training Accurary0.944291
Iter50,Testing Accurary0.9436,Training Accurary0.945236
Iter51,Testing Accurary0.9436,Training Accurary0.944891
Iter52,Testing Accurary0.9406,Training Accurary0.945764
Iter53,Testing Accurary0.9435,Training Accurary0.946764
Iter54,Testing Accurary0.9415,Training Accurary0.945255
Iter55,Testing Accurary0.9447,Training Accurary0.947273
Iter56,Testing Accurary0.9423,Training Accurary0.948309
Iter57,Testing Accurary0.9435,Training Accurary0.946964
Iter58,Testing Accurary0.9443,Training Accurary0.947618
Iter59,Testing Accurary0.9464,Training Accurary0.948
Iter60,Testing Accurary0.9425,Training Accurary0.949382
Iter61,Testing Accurary0.943,Training Accurary0.948582
Iter62,Testing Accurary0.9479,Training Accurary0.950236
Iter63,Testing Accurary0.9438,Training Accurary0.948855
Iter64,Testing Accurary0.9442,Training Accurary0.950018
Iter65,Testing Accurary0.9447,Training Accurary0.950418
Iter66,Testing Accurary0.9453,Training Accurary0.950255
Iter67,Testing Accurary0.947,Training Accurary0.949618
Iter68,Testing Accurary0.9463,Training Accurary0.950909
Iter69,Testing Accurary0.947,Training Accurary0.950545
Iter70,Testing Accurary0.9456,Training Accurary0.9518
Iter71,Testing Accurary0.9492,Training Accurary0.952164
Iter72,Testing Accurary0.9473,Training Accurary0.950945
Iter73,Testing Accurary0.9503,Training Accurary0.952655
Iter74,Testing Accurary0.9481,Training Accurary0.953091
Iter75,Testing Accurary0.9475,Training Accurary0.951782
Iter76,Testing Accurary0.9469,Training Accurary0.953382
Iter77,Testing Accurary0.9479,Training Accurary0.9528
Iter78,Testing Accurary0.9477,Training Accurary0.9524
Iter79,Testing Accurary0.9486,Training Accurary0.953927
Iter80,Testing Accurary0.9479,Training Accurary0.954927
Iter81,Testing Accurary0.9501,Training Accurary0.954382
Iter82,Testing Accurary0.9483,Training Accurary0.953455
Iter83,Testing Accurary0.9517,Training Accurary0.954691
Iter84,Testing Accurary0.9492,Training Accurary0.955327
Iter85,Testing Accurary0.9504,Training Accurary0.954836
Iter86,Testing Accurary0.9521,Training Accurary0.955182
Iter87,Testing Accurary0.9481,Training Accurary0.954691
Iter88,Testing Accurary0.9529,Training Accurary0.956818
Iter89,Testing Accurary0.9518,Training Accurary0.954745
Iter90,Testing Accurary0.9518,Training Accurary0.956509
Iter91,Testing Accurary0.9482,Training Accurary0.955545
Iter92,Testing Accurary0.9483,Training Accurary0.955982
Iter93,Testing Accurary0.9481,Training Accurary0.957091
Iter94,Testing Accurary0.9531,Training Accurary0.956364
Iter95,Testing Accurary0.9516,Training Accurary0.957364
Iter96,Testing Accurary0.9519,Training Accurary0.957582
Iter97,Testing Accurary0.95,Training Accurary0.958382
Iter98,Testing Accurary0.9519,Training Accurary0.958436
Iter99,Testing Accurary0.9505,Training Accurary0.957545
Iter100,Testing Accurary0.9521,Training Accurary0.957491
Iter101,Testing Accurary0.9506,Training Accurary0.957673
Iter102,Testing Accurary0.9525,Training Accurary0.959127
Iter103,Testing Accurary0.9507,Training Accurary0.959418
Iter104,Testing Accurary0.952,Training Accurary0.957764
Iter105,Testing Accurary0.9504,Training Accurary0.959364
Iter106,Testing Accurary0.9546,Training Accurary0.959164
Iter107,Testing Accurary0.9519,Training Accurary0.959345
Iter108,Testing Accurary0.9524,Training Accurary0.958745
Iter109,Testing Accurary0.9531,Training Accurary0.959964
Iter110,Testing Accurary0.9528,Training Accurary0.959873
Iter111,Testing Accurary0.9533,Training Accurary0.959782
Iter112,Testing Accurary0.9539,Training Accurary0.959927
Iter113,Testing Accurary0.9543,Training Accurary0.959564
Iter114,Testing Accurary0.9538,Training Accurary0.960655
Iter115,Testing Accurary0.9559,Training Accurary0.961291
Iter116,Testing Accurary0.9558,Training Accurary0.960673
Iter117,Testing Accurary0.9534,Training Accurary0.961291
Iter118,Testing Accurary0.9538,Training Accurary0.960964
Iter119,Testing Accurary0.9547,Training Accurary0.960382
Iter120,Testing Accurary0.9521,Training Accurary0.961327
Iter121,Testing Accurary0.9547,Training Accurary0.962127
Iter122,Testing Accurary0.9534,Training Accurary0.960473
Iter123,Testing Accurary0.9538,Training Accurary0.960673
Iter124,Testing Accurary0.9545,Training Accurary0.960291
Iter125,Testing Accurary0.9561,Training Accurary0.961782
Iter126,Testing Accurary0.9538,Training Accurary0.961782
Iter127,Testing Accurary0.9537,Training Accurary0.962145
Iter128,Testing Accurary0.9519,Training Accurary0.9618
Iter129,Testing Accurary0.9575,Training Accurary0.962836
Iter130,Testing Accurary0.9552,Training Accurary0.962164
Iter131,Testing Accurary0.955,Training Accurary0.962764
Iter132,Testing Accurary0.9558,Training Accurary0.963018
Iter133,Testing Accurary0.9565,Training Accurary0.963345
Iter134,Testing Accurary0.956,Training Accurary0.964218
Iter135,Testing Accurary0.9546,Training Accurary0.963345
Iter136,Testing Accurary0.9566,Training Accurary0.963273
Iter137,Testing Accurary0.9579,Training Accurary0.963818
Iter138,Testing Accurary0.9554,Training Accurary0.962891
Iter139,Testing Accurary0.9541,Training Accurary0.963127
Iter140,Testing Accurary0.9578,Training Accurary0.963182
Iter141,Testing Accurary0.957,Training Accurary0.963345
Iter142,Testing Accurary0.9573,Training Accurary0.964309
Iter143,Testing Accurary0.9572,Training Accurary0.964382
Iter144,Testing Accurary0.956,Training Accurary0.964145
Iter145,Testing Accurary0.9539,Training Accurary0.964727
Iter146,Testing Accurary0.9561,Training Accurary0.964909
Iter147,Testing Accurary0.9552,Training Accurary0.964345
Iter148,Testing Accurary0.9556,Training Accurary0.964127
Iter149,Testing Accurary0.9576,Training Accurary0.964309
Iter150,Testing Accurary0.9537,Training Accurary0.964436
Iter151,Testing Accurary0.9562,Training Accurary0.9652
Iter152,Testing Accurary0.9597,Training Accurary0.964491
Iter153,Testing Accurary0.9574,Training Accurary0.9648
Iter154,Testing Accurary0.9551,Training Accurary0.964945
Iter155,Testing Accurary0.9589,Training Accurary0.966018
Iter156,Testing Accurary0.9557,Training Accurary0.964527
Iter157,Testing Accurary0.9563,Training Accurary0.965345
Iter158,Testing Accurary0.9586,Training Accurary0.964764
Iter159,Testing Accurary0.9566,Training Accurary0.965636
Iter160,Testing Accurary0.9562,Training Accurary0.965273
Iter161,Testing Accurary0.9613,Training Accurary0.965418
Iter162,Testing Accurary0.9587,Training Accurary0.966491
Iter163,Testing Accurary0.9577,Training Accurary0.965273
Iter164,Testing Accurary0.9564,Training Accurary0.965091
Iter165,Testing Accurary0.9555,Training Accurary0.965236
Iter166,Testing Accurary0.9581,Training Accurary0.966345
Iter167,Testing Accurary0.9578,Training Accurary0.966945
Iter168,Testing Accurary0.9553,Training Accurary0.967509
Iter169,Testing Accurary0.9592,Training Accurary0.966855
Iter170,Testing Accurary0.9601,Training Accurary0.966764
Iter171,Testing Accurary0.9592,Training Accurary0.966945
Iter172,Testing Accurary0.958,Training Accurary0.966727
Iter173,Testing Accurary0.958,Training Accurary0.967382
Iter174,Testing Accurary0.9585,Training Accurary0.966691
Iter175,Testing Accurary0.9563,Training Accurary0.967964
Iter176,Testing Accurary0.9613,Training Accurary0.967455
Iter177,Testing Accurary0.9575,Training Accurary0.967873
Iter178,Testing Accurary0.9573,Training Accurary0.967927
Iter179,Testing Accurary0.9582,Training Accurary0.966964
Iter180,Testing Accurary0.9596,Training Accurary0.967436
Iter181,Testing Accurary0.9604,Training Accurary0.966982
Iter182,Testing Accurary0.9596,Training Accurary0.968036
Iter183,Testing Accurary0.9587,Training Accurary0.967491
Iter184,Testing Accurary0.9575,Training Accurary0.968327
Iter185,Testing Accurary0.9604,Training Accurary0.968364
Iter186,Testing Accurary0.9588,Training Accurary0.967945
Iter187,Testing Accurary0.9592,Training Accurary0.968036
Iter188,Testing Accurary0.9597,Training Accurary0.969055
Iter189,Testing Accurary0.9593,Training Accurary0.968345
Iter190,Testing Accurary0.9593,Training Accurary0.968418
Iter191,Testing Accurary0.9626,Training Accurary0.968873
Iter192,Testing Accurary0.9594,Training Accurary0.968927
Iter193,Testing Accurary0.9588,Training Accurary0.969436
Iter194,Testing Accurary0.9579,Training Accurary0.968964
Iter195,Testing Accurary0.9581,Training Accurary0.968127
Iter196,Testing Accurary0.961,Training Accurary0.968327
Iter197,Testing Accurary0.9593,Training Accurary0.968109
Iter198,Testing Accurary0.9622,Training Accurary0.969255
Iter199,Testing Accurary0.961,Training Accurary0.969309
'''