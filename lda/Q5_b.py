import numpy as np
import matplotlib.pyplot as plt

def get_plots(classes):
    colors = ['c', 'r']
    for i, c in zip(classes, colors):
        x = classes[i]
        x = np.array(x)
        #print(x.shape)
        #plt.scatter(x, color=c)
        val = 0. # this is the value where you want the data to appear on the y-axis.
        #ar = np.arange(10) # just as an example array
        plt.scatter(x[:,0], x[:,1], color=c)
    plt.savefig('Q5b.png')

def get_feature_map(X):
    X = np.append(X, np.power(X, 2), axis=1)
    return X

if __name__ == '__main__':
    c = 2
    n1 = n2 = 100
    xmin = [-3, 2]
    xmax = [-2, 3]
    classes = {}
    X1 = np.load('X1.npy')
    X1 = get_feature_map(X1)
    classes[0] = X1

    X2 = np.load('X2.npy')
    X2 = get_feature_map(X2)
    classes[1] = X2

    get_plots(classes)
