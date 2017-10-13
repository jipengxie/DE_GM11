#-*- coding: utf-8 -*-

def GM11(x0): #自定义灰色预测函数
    import numpy as np
    x1 = x0.cumsum() #1-AGO序列，P105
    z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列,P93,一维数组切片
    z1 = z1.reshape((len(z1),1))#表示重新建造各维度大小的元组，P370
    B = np.append(-z1, np.ones_like(z1), axis = 1)#P85ones_like(z1)以Z1为参数，根据其形状和dtype创建一个全1数组,P126
    Yn = x0[1:].reshape((len(x0)-1, 1))
    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数P97np.dot，P104，P110，(B.T*B)(-1)*B.T*Yn，-[a]称作发展系数，[b]为灰色作用量
    f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #还原值,IAGO
    delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))#残差=|原始值-还原值|
    e=delta.mean()
    C = delta.std()/x0.std()#delta.std()残差标准差，x0.std()标准差
    P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)
    print e,C,P
    print a,b
    return f, a, b, x0[0],e, C, P #返回灰色预测函数f、a、b、首项x0[0]、误差均值e、后验差比值C、平均小误差概率P
 
