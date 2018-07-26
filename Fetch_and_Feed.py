import tensorflow as tf
# Fetch:同时run多个op
input1=tf.constant(2.0)
input2=tf.constant(3.0)
input3=tf.constant(5.0)

add=tf.add(input1,input2)
mul=tf.multiply([input3],[add])

with tf.Session() as sess:
    result2=sess.run([mul,add])#[mul,add]:Fetch的妙处
    print(result2)

#Feed:为占位符喂数据
#创建占位符
input4=tf.placeholder(tf.float32)
input5=tf.placeholder(tf.float32)

output=tf.multiply(input4,input5)

with tf.Session() as sess:
    result2=sess.run(output,feed_dict={input4:[8.0],input5:[9.0]})
    print(result2)
