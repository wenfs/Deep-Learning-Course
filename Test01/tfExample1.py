import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
#学习的函数为线性函数y=2x+1

#关闭动态图
tf.disable_eager_execution()
#清除default graph和不断增加的结点  只显示接下来的操作的计算图
tf.reset_default_graph()


#==================人工数据生成===================


#设置随机数种子
np.random.seed(5)
#直接采用np生成等差数列的方法，生成100个-1~1之间的点
x_data = np.linspace(-1,1,100)
#w取2，b取1.0，加0.4的噪声 振荡  x_data.shape=(100,),*x_data.shape=(100,)=100 *将元祖拆成单独的实参
#np.random.randn(100)产生（0,1）标准正态分布的100个点,噪声的维度要与x_data一致，因为要相加
y_data = 2*x_data + 1.0 + np.random.randn(*x_data.shape)*0.4

#画出随机生成数据的散点图
plt.figure(figsize=(5,5)) #创建画布
plt.scatter(x_data,y_data)
#画出学习到的线性函数y
plt.plot(x_data,2*x_data+1.0,color='red',linewidth=3)
plt.show()

#=================构建模型=======================


#定义训练数据的占位符，X是特征值，y是标签值
x = tf.placeholder("float",name="x")
y = tf.placeholder("float",name="y")
#定义回归模型
def model(x,w,b):
    return tf.multiply(x,w)+b
#定义模型结构，有多少个参数就要定义多少个变量
#构建线性函数的斜率，变量W
w = tf.Variable(1.0,name="w0")
#构建线性函数的截距，变量b
b = tf.Variable(0.0,name="b0")
#pred节点是预测值，前向计算(通过特征值计算结果)，为的是以后随便给一个x值就会算出一个对应的y值
pred = model(x,w,b)

#设置训练参数
#迭代次数（训练论数）
train_epochs = 10
#学习率，往往是一个经验值，0.01~0.1之间
learning_rate = 0.05
#控制显示loss值的粒度
display_step = 10



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
    plt.plot(x_data,w0temp*x_data+b0temp) #总共画10条线
#图形化显示loss
plt.plot(loss_list,'r+')
plt.show()
#从loss_list中筛选出大于1的值
print("==========从loss_list中筛选出大于1的值==========")
print([x for x in loss_list if x > 1])
#打印w,b的值
print("===================打印w,b的值=================")
print("w:",sess.run(w))
print("b:",sess.run(b))

#可视化
plt.scatter(x_data,y_data,label="Original data",color="b")
plt.plot(x_data,x_data*sess.run(w)+sess.run(b),label="Fitted line",color="r",linewidth=3)
plt.legend(loc=2)#通过参数loc指定图例位置
plt.show()



#====================进行预测=====================
x_test = 3.21
predict = sess.run(pred,feed_dict={x:x_test})
#==predict = sess.run(w) * x_test +sess.run(b)
print("=======================进行预测=====================")
print("预测值：%f"%predict)



target = 2*x_test +1.0
print("真实值：%f"%target)


#步骤
# （1）生成人工数据集及其可视化
# （2）构建线性模型
# （3）定义损失函数
# （4）定义优化器、最小化损失函数
# （5）训练结果的可视化
# （6）利用学习到的模型进行预测
