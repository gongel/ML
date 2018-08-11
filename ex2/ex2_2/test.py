import numpy as np
# b=np.array([[1],[2],[3],[4]])
# # print(a)
# # print(a-1)
# # print(len(a))
# # b=np.mat([1,2,3,4])
# # print(b[0,1])
#
# import pandas as pd
# path='./ex2data2.txt'
# data=pd.read_csv(path)
# cols=data.shape[1]
# y1=data.iloc[:,0:1]
# # print('y1**************',y1)
# # y2=data.iloc[:,0:]
# # print('y2**************',y2)#      y1!=y2
#
# c=np.matrix([4])
# print(c)
# # print(c.getA()[0][0])#矩阵内积还是矩阵，这是取值
#
# import numpy as np
# a=np.matrix([[1],[2],[3],[4]])
# print(a-1)
# '''
# [[0]
#  [1]
#  [2]
#  [3]]
#  '''
# b=np.array([[1],[2],[3],[4]])
# print(b+1)
# '''
# [[2]
#  [3]
#  [4]
#  [5]]
#  '''
#
# print(b.ravel())
# print(a.ravel())

a=np.matrix([[1],[2],[3],[4]])
print(a)
print(a-1)

grad=np.zeros(5)
print(grad[0][1])