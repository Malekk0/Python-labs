import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import math


def sigmoid(x):
  return 1/(1+math.exp(-x))



file = r'pima-indians-diabetes.csv'
df = pd.read_csv(file)

data = df.values

X=data[:,:9]
Y=data[:,8:]*2-1
l=Y.shape[0]
X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)



w=np.zeros(10)

n=0.1

for k in range(500):
  sum=0
  for j in range(10):
    for i in range (l):
      sum+=Y[i]*X[i,j]*sigmoid(-Y[i]*w.dot(X[i]))

  w[j]=w[j]+n*(1.0/l)*sum

y_pr=X.dot(w)
y_pr_b=[]

for x in y_pr:
  y_pr_b.append(1 if x<0 else 0)


print(y_pr_b)

sum=0

for i in range(l):
  sum+=1 if Y[i]==y_pr_b[i]*2-1 else 0

A=sum/l
print(A)