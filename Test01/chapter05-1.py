import numpy as np

np.random.seed(612)
p = np.random.uniform(0,1,1000)
num = int(input("请输入一个1-100间的整数："))
a=1
for i in range(0,1000):
    if i%num == 0:
        print("%d %d %f"%(a,i,p[i]))
        a+=1




