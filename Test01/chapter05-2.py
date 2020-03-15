import numpy as np

x=[64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03]
y=[62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84]
aveX = sum(x)/10
aveY = sum(y)/10
sum1=sum2=0
for i in range(10):
    sum1+=(x[i]-aveX)*(y[i]-aveY)
for j in range(10):
    sum2+=(x[i]-aveX)**2
w=sum1/sum2
b=aveY-w*aveX
print("W的值为：%d"%w)
print("b的值为：%d"%b)