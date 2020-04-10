import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import  shuffle
from sklearn.preprocessing import scale

#========================读取数据=========================
#通过pandas导入数据，header=0表示：第1行数据作为列标题
df = pd.read_csv("data/boston.csv",header=0)
#数据探索
print(df.describe())
#df.head(3) #显示前3条数据
#df.tail(3)  #显示后3条数据
#获取数据集的值 , df.values以np.array形式返回数据集的值
ds = df.values
#print(ds.shape)--查看数据的形状（506,13）
#print(ds) #查看数据集的值

#========================划分特征数据和标签数据=======================
x_data = ds[:,:12]
y_data = ds[:,12]
#特征数据归一化
for i in range(12):
    x_data[:,i] = (x_data[:,i]-x_data[:,i].min())/(x_data[:,i].max()-x_data[:,i].min())

train_num = 300
valid_num = 100   #验证集数目
test_num = len(x_data) -train_num - valid_num

#取出训练集
x_train = x_data[:train_num]
y_train = y_data[:train_num]
#取处验证集
x_valid = x_data[train_num:train_num+valid_num]
y_valid = y_data[train_num:train_num+valid_num]
#取处测试集
x_test = x_data[train_num+valid_num:train_num+valid_num+test_num]
y_test = y_data[train_num+valid_num:train_num+valid_num+test_num]

#转换数据类型
x_train = tf.cast(x_train,dtype=tf.float32)
x_valid = tf.cast(x_valid,dtype=tf.float32)
x_test = tf.cast(x_test,dtype=tf.float32)
#也可在此处scale(对原始数据归一)化处理
# x_train = tf.cast(scale(x_train),dtype=tf.float32)
# x_valid = tf.cast(scale(x_valid),dtype=tf.float32 )
# x_test = tf.cast(scale(x_test),dtype=tf.float32) 




#==========================构建模型====================
#定义模型
def model(x,w,b):
    return tf.matmul(x,w)+b
#创建待优化变量
W = tf.Variable(tf.random.normal([12,1],mean=0.0,stddev=1.0,dtype=tf.float32))
B = tf.Variable(tf.zeros(1),dtype = tf.float32)


#==========================模型训练===========================
#设置超参数
training_epochs = 50
learning_rate = 0.001
batch_size = 10 #批量训练一次的样本数
#定义均方差损失函数
def loss(x,y,w,b):
    err = model(x,w,b) - y    #计算模型预测值和标签值的差异
    squared_err = tf.square(err)     #求平方，得出方差
    return tf.reduce_mean(squared_err)   #求均值，得出均方差
#定义梯度计算函数
#计算样本数据【x,y】在参数【w，b】点上的梯度
def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_ = loss(x,y,w,b)
    return tape.gradient(loss_,[w,b]) #返回梯度向量
#选择优化器
optimizer = tf.keras.optimizers.SGD(learning_rate)  #帮助根据计算出的求导结果更新模型参数，从而最小化损失函数




#===========================迭代训练==========================
loss_list_train = []
loss_list_valid = []
total_step = int(train_num/batch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = x_train[step*batch_size:(step+1)*batch_size,:]
        ys = y_train[step*batch_size:(step+1)*batch_size]

        #每训练10条数据进行一次梯度计算和优化w,b
        grads = grad(xs,ys,W,B)  #计算梯度
        optimizer.apply_gradients(zip(grads,[W,B]))
    
    loss_train = loss(x_train,y_train,W,B).numpy() #计算当前轮训练损失
    #验证集不用进行训练，用于验证
    loss_valid = loss(x_valid,y_valid,W,B).numpy() #计算当前轮验证损失
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d},train_loss={:.4f},valid_loss={:.4f}".format(epoch+1,loss_train,loss_valid))


#==========================可视化损失值==============================
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train,'b',label="Train Loss")
plt.plot(loss_list_valid,'r',label="Valid Loss")
plt.legend(loc=1)#参数loc指定参数位置
plt.show()
#查看测试集的损失
print("Test_loss:{:.4f}".format(loss(x_test,y_test,W,B).numpy()))

#===========================模型应用===============================
test_house_id = np.random.randint(0,test_num)
y = y_test[test_house_id]
y_pred = model(x_test,W,B)[test_house_id]
y_predit=tf.reshape(y_pred,()).numpy()
print("House id",test_house_id,"Actual value",y,"Predicted value",y_predit)
