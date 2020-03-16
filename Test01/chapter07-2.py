import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(5,5))

mnist = tf.keras.datasets.mnist
(tra_x,tra_y),(tes_x,tes_y)=mnist.load_data()
train_x=np.array(tra_x)
train_y=np.array(tra_y)
test_x=np.array(tes_x)
test_y=np.array(tes_y)


for i in range(16):
    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(tra_x[i],cmap="gray")
    plt.title("标签值:"+str(train_y[i]),fontsize=14)

plt.suptitle("MNIST测试集样本",fontsize=20,color="red")
plt.tight_layout(rect=[0.0,0.0,1.0,0.9]) #调整子图间距
plt.show()