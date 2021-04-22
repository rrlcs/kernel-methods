import numpy as np
import matplotlib.pyplot as plt

def get_plots(classes, W, X):
    colors = ['c', 'r']
    for i, c in zip(classes, colors):
        x = classes[i]
        x = np.array(x)
        plt.scatter(x[:,0], x[:,1], color=c)
    b = 2.
    ar = np.arange(200)
    xpoints = [-W[0], W[0]]
    ypoints = [-W[1], W[1]]
    plt.plot(xpoints , ypoints, color='g')
    plt.savefig('Q5c.png')

def get_feature_map(X):
    X = np.append(X, np.power(X, 2), axis=1)
    return X

def get_S_within(classes):
    m = []
    for i in classes:
        x = classes[i]
        x = np.array(x)
        n = len(x)
        m.append(np.mean(x, axis=0))
    m = np.array(m).reshape(2,2)

    S = {}
    for i in classes:
        x = classes[i]
        x = np.array(x)
        t1 = x-m[i,:]
        t2 = x-m[i,:].T
        S[i]  = np.dot(t1.T,t2)
    SW = S[0] + S[1]

    SWI = np.linalg.inv(SW)
    m1m2 = m[0,:] - m[1,:]
    W = np.dot(SWI, m1m2).reshape(2,1)
    return W

def LDA(classes, X):
    W = get_S_within(classes)
    d = W/np.linalg.norm(W)
    W = d*10
    get_plots(classes, W, X)

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
    X = np.vstack((X1, X2))
    LDA(classes, X)
