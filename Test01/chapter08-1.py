import numpy as np
import tensorflow as tf
x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
#创建张量
x1=tf.constant(x)
y1=tf.constant(y)
#求张量均值
aveX = tf.reduce_mean(x1)
aveY = tf.reduce_mean(y1)

sum1 = tf.reduce_sum((x1-aveX)*(y1-aveY))
sum2 = tf.reduce_sum((tf.pow((x1-aveX),2)))

w=sum1/sum2
b=aveY-w*aveX

print("W的值为：%f"%w)
print("b的值为：%f"%b)