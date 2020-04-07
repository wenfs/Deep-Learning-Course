import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
#波士顿房价

#关闭动态图
tf.disable_eager_execution()
#清除default graph和不断增加的结点  只显示接下来的操作的计算图
tf.reset_default_graph()

#==================数据准备=================
#读取数据文件
df = pd.read_csv("data/boston.csv")
#显示数据摘要描述信息
#print(df.describe())
#载入本示例所需数据
df = df.values
#把df转换为np数组格式
df = np.array(df)

#对特征数据【0-11】列做归一化处理
for i in range(12):
    df[:,i] = (df[:,i]-df[:,i].min())/(df[:,i].max()-df[:,i].min())


#X_data为前12列特征数据
x_data = df[:,:12]
#y_data为最后1列标签数据
y_data = df[:,12]


#====================模型定义===================
#定义训练数据占位符
#定义特征数据和标签数据的占位符
#shape中的None表示行的数量未知，在实际训练时决定一次代入多少行样本
x = tf.placeholder(tf.float32,[None,12],name="X")#12个特征数据（12列）
y = tf.placeholder(tf.float32,[None,1],name="Y")#1个标签数据（1列）
#定义模型函数
#定义了一个命名空间Model,相当于打包
with tf.name_scope("Model"):
    #w初始化值为shape=(12,1)的随机数
    w = tf.Variable(tf.random_normal([12,1],stddev=0.01),name="W")
    #b初始化值为1.0
    b = tf.Variable(1.0,name="b")
    def model(x,w,b):
        return tf.matmul(x,w)+b #w和x是矩阵相乘
    #预测计算操作，前向计算节点
    pred = model(x,w,b)



#=====================模型训练======================
#设置训练超参数
train_epochs = 50
#学习率
learning_rate = 0.01

#定义均方差损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred,2)) #均方误差

#选择优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_function)

#声明会话
sess = tf.Session()
#定义初始化变量的操作
init = tf.global_variables_initializer()

#为TensorBoard可视化准备数据
logdir='D:\pycharm2019\Test01'
#创建一个操作，用于记录损失值loss,后面再TensorBoard中的SCALARS栏可见
sum_loss_op = tf.summary.scalar("loss",loss_function)
#把所有需要记录的摘要日志文件进行合并，方便一次性写入
merged = tf.summary.merge_all()


#启动会话
sess.run(init)

#创建摘要的文件写入器（FileWritter）
#创建摘要writer,将计算图写入摘要文件，后面再Tensorboard中的GRAPHS栏可见
writer = tf.summary.FileWriter(logdir,sess.graph)

#迭代训练
#可视化损失值
loss_list = []
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs,ys in zip(x_data,y_data):
        #Feed数据必须和placeholder的shape一致
        xs = xs.reshape(1,12)
        ys = ys.reshape(1,1)
        _,summary_str,loss = sess.run([optimizer,sum_loss_op,loss_function],feed_dict={x:xs,y:ys})
        writer.add_summary(summary_str,epoch)

        loss_sum = loss_sum + loss
        #loss_list.append(loss) #每步操作添加一次
    #打乱数据顺序
    x_data,y_data = shuffle(x_data,y_data)

    b0temp = b.eval(session=sess)  # 进行一轮训练完成后，得出w和b个值，画一条直线
    w0temp = w.eval(session=sess)
    loss_average = loss_sum/len(y_data)

    #loss_list.append(loss_average) #每轮将损失平均值加入列表
    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)

#======================模型应用=========================
n = np.random.randint(506)
print(n)
x_test = x_data[n]
x_test = x_test.reshape(1,12)
predict = sess.run(pred,feed_dict={x:x_test})
print("预测值：%f"%predict)
target = y_data[n]
print("标签值：%f"%target)
