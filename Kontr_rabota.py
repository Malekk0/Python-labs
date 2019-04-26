import numpy as np
import pandas as pd




file = r'Consumo_cerveja.csv'
df = pd.read_csv(file)

data = df.values



for i in range(0, data.shape[0]):
    data[i, 0] = i



X=data[:,0]
Y=data[:,1:len(data)]



X=np.array(X)
Y=np.array(Y)


print(X)
