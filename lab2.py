import numpy as np
import matplotlib.pyplot as plt


X = []
Y = []



for line in open('India GDP from 1961 to 2017.csv'):
    x, y = line.split(',')
    X.append(int(x))
    Y.append(float(y))



X = np.array(X)
Y = np.array(Y)

X_mean = X.mean()
X2_mean = (X * X).mean()
Y_mean = Y.mean()
XY_mean = (X * Y).mean()

denominator = X2_mean - X_mean**2
a = (XY_mean - X_mean * Y_mean) / denominator
b = (Y_mean * X2_mean - X_mean * XY_mean) / denominator

Y_predict = a*X+b

plt.scatter(X, Y, color="red")
plt.plot(X, Y_predict, color="blue")
plt.show()