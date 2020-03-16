import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(5,5))
image = Image.open("lena.tiff")
(img_r,img_g,img_b)=image.split()

plt.subplot(2,2,1)
plt.axis("off")
img_r_small=img_r.resize((50,50))
plt.imshow(img_r_small,cmap="gray")
plt.title("R-缩放",fontsize=14)
#plt.show()

plt.subplot(2,2,2)
img_r_rot=img_g.transpose(Image.FLIP_LEFT_RIGHT) #水平镜像
img_r_ro=img_r_rot.transpose(Image.ROTATE_270) #顺时针90
plt.imshow(img_r_ro,cmap="gray")
plt.title("G-镜像+旋转",fontsize=14)

plt.subplot(2,2,3)
plt.axis("off")
img_r_cut=img_b.crop((0,0,150,150))
plt.imshow(img_r_cut,cmap="gray")
plt.title("B-裁剪",fontsize=14)

plt.subplot(2,2,4)
plt.axis("off")
img_rgb=Image.merge("RGB",[img_r,img_g,img_b])
plt.imshow(img_rgb)
plt.title("RGB",fontsize=14)

plt.suptitle("图像基本操作",fontsize=20,color="blue")
plt.tight_layout(rect=[0.0,0.0,1.0,0.9]) #调整子图间距
plt.show()
