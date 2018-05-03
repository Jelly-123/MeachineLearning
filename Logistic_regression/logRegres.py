#coding:utf-8
import random
import numpy as np
from math import exp
from numpy import mat
from numpy import ones
from numpy import shape
#打开文件testSet.txt,每行前两个值分别是X1和X2，第三个值是数据对应的类别标签，将X0设置为x1
def loadFile():
    fp = open('testSet.txt')
    lines = fp.readlines()
    data=[]
    label=[]
    for line in lines:
        line = line.strip()
        dataOfLine = line.split('\t')
        data.append([1,float(dataOfLine[0]),float(dataOfLine[1])])
        label.append(float(dataOfLine[2]))
    return data,label

def sigmod(intX):
    return 1.0/(1+np.exp(-intX))

def gradAscent(dataMatIn,classLabels):
    datamatrix=mat(dataMatIn)
    classLabels = mat(classLabels).T
    m,n=shape(datamatrix)
    alpha=0.001
    maxcycles=500
    weights=ones((n,1))

    for k in range(maxcycles):
        h=sigmod(datamatrix*weights) 
        #这里的datamatrix*weights是sigmod函数的输入，默认weights都为1
        error=classLabels-h
        weights=weights+alpha*datamatrix.T*error
        #这里的推导公式见本本
    print("the weights is:",weights)
    return weights
    #返回了一组回归系数
#随机梯度上升算法:梯度上升法在循环500次选择最优的梯度时，每次要计算输入数据与系数的乘机，即是300×500次
#倘若数据比较大的时候，计算量也会变大，故在优化系数时，我们只做一部分的乘积
#input:data=array(data) 参数data:array label:list
def stocGradAscent(data,label):
    m,n=shape(data)
    alpha=0.01
    #这里的步长很重要，影响最佳拟合直线的走向
    weights=ones(n)
    for i in range(m):
        h=sigmod(sum(data[i]*weights))
        error=label[i]-h
        weights=weights+alpha*error*data[i]
       # A,B是array类型的，不是matrix，所以不能使用矩阵的乘法。方法1.使用array中numpy中的dot函数，实现两个二维数组的乘积2.将数组转为矩阵
    print("the stocGradAscent weights is:",weights)
    return weights
#对上面算法的改进，改变周期性波动引入随机 每次迭代都调整步长，这里不太明白
def stocGradAscent1(data,label,num=150):
    m,n=shape(data)
    weights=ones(n)
    for j in range(num):
        dataIndex=range(m)
        #type(dataIndx);l
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randomNum=int(random.uniform(0,len(dataIndex)))
            h=sigmod(sum(data[randomNum]*weights))
            error=label[randomNum]-h
            weights=weights+alpha*error*data[randomNum]
            del(dataIndex[randomNum])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    data,label=loadFile()
    dataArr = np.array(data)
    datashape=shape(dataArr)
    print("the shape of dataArr :",datashape)
    n=datashape[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(label[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    #画出直线,weights[0]*1+weights[1]*x+weights[2]*y=0
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()
