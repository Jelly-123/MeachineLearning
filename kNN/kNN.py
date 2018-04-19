#coding:utf-8
from numpy import *
import operator
import math

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify(intX,group,labels,k):
    count_group = len(group)
    result_list = []
    for i in range(count_group):
        jiashu_one = group[i][0]-intX[0]
        jiashu_one = pow(jiashu_one,2)
        jiashu_two = group[i][1]-intX[1]
        jiashu_two = pow(jiashu_two,2)
        he = jiashu_one + jiashu_two
        result = sqrt(he)
        result_list.append(result)
    return result_list
    #type(result_list):list

def classify_2(intX,group,labels,k):
    dataSetSize = group.shape[0]
    #求group的行号
    diffMat = tile(intX,(dataSetSize,1))-group
    #tile(A，（2,1）)将A复制成两行一次的矩阵，再和原数据相减
    sqDiffMat = diffMat **2
    sqDistance = sqDiffMat.sum(axis =1) 
    #axis=1按照横轴，sum表示累加，即按照行进行累加
    distance = sqDistance ** 0.5
    #type(distance):numpy.ndarry
    sortedDistIndicies =distance.argsort()
    #type(sortedDistIndicies):numpy.ndarray
    #快排由小到大，原下标
    classCount={}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse =True)
    #items能够得到一个关于字典的列表，列表中的元素是由字典中的键和值所组成的元组，表示用第二域来排序
    #sorted()第一个参数：指定要排序的list.第二个参数为指定排序时进行比较的函数，reverse=True 是降序
    return sortedClassCount[0][0]

if __name__=="__main__":
    group,labels=createDataSet()
    intX= [0.1,0.1]
    className = classify_2(intX,group,labels,3)
    print('the clas of test example is %s' %className)

