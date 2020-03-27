import numpy as np
import tensorflow as tf
#numpy数组输入
x1=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
x2=np.array([3.,2.,2.,3.,1.,2.,3.,2.,2.,3.,1.,1.,1.,1.,2.,2.])
y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
x0=np.ones(len(x1))
#转换为张量
x11=tf.constant(x1)
x22=tf.constant(x2)
yy=tf.constant(y)
x00=tf.constant(x0)

X=tf.stack((x00,x11,x22),axis=1)
Y=tf.reshape(yy,[-1,1])
Xt=tf.transpose(X,perm=[1,0])  #计算转置
XtX_1=tf.linalg.inv(tf.matmul(Xt,X)) #求逆
XtX_1_Xt=tf.matmul(XtX_1,Xt)

W=tf.matmul(XtX_1_Xt,Y)
W=tf.reshape(W,[-1])
print("多元线性回归方程：")
print("Y=",W[1].numpy(),"*x1+",W[2].numpy(),"*x2+",W[0].numpy())
#面积输入的规则
def inputArea():
    try:
        X1_test = float(input("商品房面积(20-500之间的实数)："))
        if X1_test <20 or X1_test>500:
            return 1
        return X1_test
    except:
        return 0
#房间数输入的规则
def inputNum():
    try:
        X2_test=int(input("房间数(1-10之间的整数):"))
        if X2_test<1 or X2_test>10:
            return -1
        return X2_test
    except:
        return 0

X1_test=inputArea()
while X1_test==0 or X1_test==1:
    if X1_test==0:
        print("您输入的类型有错")
    else:
        print("您输入的范围有错")
    X1_test = inputArea()
X2_test=inputNum()
while X2_test==0 or X2_test==-1:
    if X2_test==0:
        print("您输入的类型有错")
    else:
        print("您输入的范围有错")
    X2_test=inputNum()

y_pred=W[1]*X1_test+W[2]*X2_test+W[0]
print("根据您的输入，预测的价格为：",y_pred.numpy().round(2),"万元")