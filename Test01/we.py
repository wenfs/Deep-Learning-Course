import tensorflow
import cmath

a = int(input("请输入任意a值："))
b = int(input("请输入任意b值："))
c = int(input("请输入任意c值："))
print("您输入的a,b,c为：%d,%d,%d"%(a,b,c))
da=b**2-4*a*c
if da < 0 :
    print("该方程在您设置的参数下无解")
elif da > 0:
    print("该方程有两个解！")
    x1=(-b+cmath.sqrt(da))/(2*a)
    x2=(-b-cmath.sqrt(da))/(2*a)
    print("x1和x2分别为：{} {}".format(x1,x2))
else:
    print("该方程有且仅有一个解")
    print("x的值为：%d"%(-b/(2*a)))


