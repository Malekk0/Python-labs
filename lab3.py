import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math


def sigmoid(x):
  if x < 0:
    return 1 - 1/(1 + math.exp(x))
  else:
    return 1/(1 + math.exp(-x))



file = r'Breast_cancer_data.csv'
df = pd.read_csv(file)

data = df.values

X=data[:,:5]
Y=data[:,5:]*2-1

l = Y.shape[0]

X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

w = np.random.normal(0.0, 1.0, 6)

n = 0.2
for k in range(200):
    for j in range(6):
        sum = 0
        for i in range(l):
            sum += Y[i] * X[i, j] * sigmoid(-Y[i] * w.dot(X[i]))

        w[j] = w[j] + n * (1.0 / l) * sum

y_predict = X.dot(w)
y_predict_b = []

for x in y_predict:
    y_predict_b.append(1 if x > 0 else -1)

sum = 0
for i in range(l):
    sum += 1 if Y[i] == y_predict_b[i] else 0

A = sum / l
print(A)

