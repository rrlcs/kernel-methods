import dill
import numpy as np

with open('k5sampler.pkl', 'rb') as in_strm:
    in_strm = open('k5sampler.pkl', 'rb')
data = dill.load(in_strm)
k5sampler = dill.loads(data)
X = []
for i in range(5):
    x = k5sampler()
    X.append(x)
X = np.array(X)
X = X[:,:,0].T
#X = X.T
print(X.shape)
print(X)
