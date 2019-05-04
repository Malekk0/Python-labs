import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




file = r'Consumo_cerveja.csv'
df = pd.read_csv(file)

data = df.values



for i in range(0, data.shape[0]):

    data[i, 0] = i

X=data[:,:1].astype(float)
Y=data[:,1:].astype(float)

X = np.array(X)
Y = np.array(Y)



r = np.linalg.solve(X.T.dot(X), X.T.dot(Y))


Y_pr = X.dot(r)

R_2 = 1 - (((Y - Y_pr)**2).sum()) / (((Y - Y_pr.mean())**2).sum())

print(R_2)

plt.scatter(Y, Y_pr)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], c='r')
plt.show()
plt.hist(X)

