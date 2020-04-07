import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
#学习的函数为线性函数y=3.1234*x+2.98

#关闭动态图
tf.disable_eager_execution()
#清除default graph和不断增加的结点  只显示接下来的操作的计算图
tf.reset_default_graph()


#==================人工数据生成===================

np.random.seed(5)
x_data = np.linspace(0,100,500)
y_data = 3.1234*x_data + 2.98 + np.random.randn(*x_data.shape)*0.4
#画出随机生成数据的散点图
plt.figure(figsize=(5,5)) #创建画布
plt.scatter(x_data,y_data)
#画出学习到的线性函数y
plt.plot(x_data,2*x_data+1.0,color='red',linewidth=3)

#清空图并为tensorboard指定生成图位置
tf.reset_default_graph()
logdir="D:\pycharm2019\Test01"

#标准化
mean_value=np.mean(x_data,axis=0)
sigma=np.std(x_data,axis=0)
x_data=(x_data-mean_value)/sigma





#=================构建模型=======================


#定义训练数据的占位符，X是特征值，y是标签值
x = tf.placeholder("float",name="x")
y = tf.placeholder("float",name="y")
#定义回归模型
def model(x,w,b):
    return tf.multiply(x,w)+b
w = tf.Variable(2.0,name="w0")
b = tf.Variable(0.0,name="b0")
pred = model(x,w,b)

#设置训练参数
#迭代次数（训练论数）
train_epochs = 20
#学习率，往往是一个经验值，0.01~0.1之间
learning_rate = 0.1
#控制显示loss值的粒度
display_step = 20



#=================定义损失函数====================

#采用均方差（L2）作为损失函数,y为真值，pred为预测值
loss_function = tf.reduce_mean(tf.square(y-pred))
#定义优化器
#梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_function)



#===================创建会话======================

#声明会话
sess = tf.Session()
#变量初始化（将所有变量初始化）
init = tf.global_variables_initializer()
sess.run(init)

#=====================迭代训练===================
#模型训练阶段，设置迭代轮次，每次通过将样本逐个输入模型，进行梯度下降优化操作
#每轮迭代后，绘制出模型曲线

#开始训练，轮数为epoch,采用SGD随机梯度下降优化算法
step = 0 #记录训练步数
loss_list = [] #用于保存loss值的列表
for epoch in range(train_epochs):
    for xs,ys in zip(x_data,y_data):
        _,loss=sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        #显示损失值loss
        #display_step：控制报告的粒度。隔display_step报告一次
        #与超参数不同，修改dispaly_step的值不会影响模型学习的规律
        loss_list.append(loss)
        step=step+1
        if step%display_step == 0:
            print("Train Epoch:",'%02d'%(epoch+1),"Step:%03d"%(step),"loss={:.9f}".format(loss))
    b0temp=b.eval(session=sess) #进行一轮训练完成后，得出w和b个值，画一条直线
    w0temp=w.eval(session=sess)
    w_origin0 = w0temp / sigma
    b_origin0 = b0temp - w0temp * mean_value / sigma
    print("===================打印w,b的原始值=================")
    print(w0temp, b0temp)
    print("===================打印w,b的标准值=================")
    print(w_origin0, b_origin0)
    plt.plot(x_data * sigma + mean_value, w0temp * x_data + b0temp)
plt.show()







#====================进行预测=====================
x_test = 5.79
predict = sess.run(pred,feed_dict={x:(x_test-mean_value)/sigma})
#==predict = sess.run(w) * x_test +sess.run(b)
print("=======================进行预测=====================")
print("预测值：%f"%predict)

target = 3.1234*x_test +2.98
print("真实值：%f"%target)

#保存图
writer=tf.summary.FileWriter(logdir,tf.get_default_graph())
writer.close()
