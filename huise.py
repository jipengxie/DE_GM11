#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from DE_GM11 import DE_GM11
from GGM import GM11
inputfile = 'D:/CGI/python/spyder/luxury/AQI.csv'
outputfile = 'D:/CGI/python/spyder/luxury/AQI1.xls'
data = pd.read_csv(inputfile)


data.index = range(0, len(data))

data.loc[len(data)] = None
data.loc[len(data)] = None
l = ['x1']

for i in l:
  f = DE_GM11(data[i][range(0, len(data)-2)].as_matrix())[0]
  data[i][len(data)-2] = f(len(data)-1) 
  data[i][len(data)-1] = f(len(data)) 
  data[i] = data[i].round(2) 
print data

