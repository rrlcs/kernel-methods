import numpy
X = numpy.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1)
X = X.reshape(100,2)
print(X.shape)
