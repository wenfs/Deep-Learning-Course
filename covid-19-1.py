# import the necessary packages
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

train_path = '../input1/chest-xray-pneumonia/chest_xray/train'
test_path='../input1/chest-xray-pneumonia/chest_xray/test'


#列出路径下的文件名或图片名并且存入list列表，进行for循环取出，构建绝对路径即可对该文件进行遍历操作。
train_imagePaths = list(paths.list_images(train_path))
test_imagePaths = list(paths.list_images(test_path))




#================================Data preprocessing=====================
# 设置超参数learning rate、epochs、batch size
INIT_LR = 1e-3
EPOCHS = 5
BS = 8

# 获取数据集目录中的图像列表，然后初始化
# # the list of data (i.e., images) and class images
print("[INFO] loading images...")
data = []
labels = []
# 循环图片路径
for imagePath in train_imagePaths:
    # 从文件名中提取类标签
    label = imagePath.split(os.path.sep)[-2]
    # 加载图片, swap color channels, and resize it to be a fixed
    # 224x224 pixels 忽略长宽比
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # 分别更新数据和标签列表
    data.append(image)
    labels.append(label)
# 在缩放像素时，将数据和标签转换为NumPy数组
# 强度范围[0,1]（归一化）
data = np.array(data) / 255.0
labels = np.array(labels)

data1 = []
labels1 = []
# 循环图片路径
for imagePath in test_imagePaths:
    # 从文件名中提取类标签
    label = imagePath.split(os.path.sep)[-2]
    # 加载图片, swap color channels, and resize it to be a fixed
    # 224x224 pixels 忽略长宽比
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # 分别更新数据和标签列表
    data1.append(image)
    labels1.append(label)
# 在缩放像素时，将数据和标签转换为NumPy数组
# 强度范围[0,1]（归一化）
data1 = np.array(data) / 255.0
labels1 = np.array(labels)



#对标签执行一次热编码
lb = LabelBinarizer()
#训练集标签
labels = lb.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)
#测试集标签
labels1 = lb.fit_transform(labels1)
labels1 = tf.keras.utils.to_categorical(labels1)
# 初始化训练数据扩充对象
trainAug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, fill_mode="nearest")


#=======================Model===================================
#加载VGG16网络，确保关闭head FC层集
baseModel = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
#构造模型的头部，它将被放置在基本模型的顶部
headModel = baseModel.output

headModel = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
headModel = tf.keras.layers.Dense(64, activation="relu")(headModel)
headModel = tf.keras.layers.Dropout(0.5)(headModel)
headModel = tf.keras.layers.Dense(2, activation="softmax")(headModel)
#将头部FC模型放在基础模型的顶部(这将成为我们训练的实际模型)
model = tf.keras.models.Model(inputs=baseModel.input, outputs=headModel)
#循环基础模型中的所有层，并冻结它们，这样它们就不会在第一个训练过程中被更新
for layer in baseModel.layers:
    layer.trainable = False

model.summary()
#
#========================Training====================
# compile our model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(data, labels, batch_size=BS),
    steps_per_epoch=len(data) // BS,
    validation_data=(data1, labels1),
    validation_steps=len(data1) // BS,
    verbose=2,
    epochs=EPOCHS)


#=============================Plot trining metrics=================================
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()




#=================================测试============================
# make predictions on the testing set
print("[INFO] testing network...")
predIdxs = model.predict(data1, batch_size=BS)
#对于测试集中的每一张图像，我们需要找到对应的预测概率最大的标签索引,if[0.23,0.77],返回1，表示属于第二类
#返回最大数索引，行方向上最大数索引，是一个二维矩阵
predIdxs = np.argmax(predIdxs, axis=1)
print(predIdxs.shape)
#显示一个格式良好的分类报告
print(classification_report(labels1.argmax(axis=1), predIdxs, target_names=lb.classes_))







#===============================Confusion matrix==================================
#计算混淆矩阵，并使用它来推导原始的准确性、敏感性和特异性
cm = confusion_matrix(labels1.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, 敏感性和特异性
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))
