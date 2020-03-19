import numpy as np
import tensorflow as tf

x=np.array([ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
y=np.array([ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])
#创建数组张量
a = tf.constant(x)
b = tf.constant(y)

f1 = len(a)*tf.reduce_sum(a*b)
f2 = tf.reduce_sum(a)*tf.reduce_sum(b)
f3 = len(a)*tf.reduce_sum(tf.pow(a,2))
f4 = tf.pow(tf.reduce_sum(a),2)
w = (f1-f2)/(f3-f4)
print("w = ",tf.cast(w,tf.float32).numpy())

f11=tf.reduce_sum(b)-w*tf.reduce_sum(a)
b = f11/len(a)
print("b = ",tf.cast(b,tf.float32).numpy())


