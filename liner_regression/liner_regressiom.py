#coding=UTF-8
import pandas as pd
from io import StringIO

from sklearn import linear_model

import matplotlib.pyplot as plt

csv_data = unicode('square_feet,price\n150,6450\n200,7450\n250,8450\n300,9450\n350,11450\n400,15450\n600,18450\n')

df = pd.read_csv(StringIO(csv_data))
print(df)

#建立线性回归模型,sklearn
regr = linear_model.LinearRegression()

#拟合 fit(X,Y),对训练集X，Y进行训练
regr.fit(df['square_feet'].reshape(-1,1),df['price'])

#不难得到直线的斜率、截距,对于线性回归问题计算得到的feature的系数。intercept:线性模型中的独立项
a,b=regr.coef_,regr.intercept_

print(a)
print(b)
#给出待预测面积

area = 238.5
#根据直线方程计算价格
print(a*area+b)

#根据predict方法预测价格
print(regr.predict(area))

#画图
#1.真实的点
plt.scatter(df['square_feet'],df['price'],color='blue')
#2.拟合的数据
plt.plot(df['square_feet'],regr.predict(df['square_feet'].reshape(-1,1)),color='red',linewidth =4)

plt.show()
