import matplotlib.pyplot as plt
import tensorflow as tf


# 加载数据集
boston_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(_,_) = boston_housing.load_data(test_split=0)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

titles=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B-1000","LSTAT","MEDV"]
plt.figure(figsize=(12,12)) #创建画布
for i in range(13):
    plt.subplot(4,4,(i+1)) #划分子图
    plt.scatter(train_x[:,i],train_y,s=10) #画散点图
    plt.ylim(0,80)
    plt.xlabel(titles[i])
    plt.ylabel("Prices($1000's)")
    plt.title(str(i+1)+"."+titles[i]+" - Price",fontsize=10)

plt.tight_layout(pad=0.90,w_pad=0.56,h_pad=3.50,rect=[0.25,0.12,0.77,0.90]) #调整子图间距
plt.suptitle("各个属性与房价的关系",x=0.5,fontsize=14)#全局标题
plt.show()

#显示1-13的提示信息
for i in range(13):
    print(str(i+1)+" -- "+titles[i])
num = int(input("请选择属性：")) #接受用户输入的num
plt.scatter(train_x[:, num-1], train_y, s=20)
plt.ylim(0, 80)
plt.xlabel(titles[num-1])
plt.ylabel("Prices($1000's)")
plt.title(str(num) + "." + titles[num-1] + " - Price", fontsize=10)
plt.show()



