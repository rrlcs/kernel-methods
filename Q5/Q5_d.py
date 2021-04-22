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
    plt.savefig('Q5d.png')

def kernel_function(x, y, sigma=3.8):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def get_M1_M2(X, X1, X2):
    M1 = []
    M2 = []
    for j in range(len(X)):
        s = 0
        for k in range(len(X1)):
            s = s + kernel_function(X[j], X1[k])
        M1.append(s/len(X1))
        for k in range(len(X2)):
            s = s + kernel_function(X[j], X2[k])
        M2.append(s/len(X2))
    M1 = np.array(M1).reshape(-1,1)
    M2 = np.array(M2).reshape(-1,1)
    return M1, M2

def get_N(X, X1, X2):
    K1 = np.zeros((len(X), len(X1)))
    K2 = np.zeros((len(X), len(X2)))
    for n in range(len(X)):
        for m in range(len(X1)):
            K1[n,m] = kernel_function(X[n], X1[m])
            K2[n,m] = kernel_function(X[n], X2[m])
    I1 = np.identity(len(X1))
    l1 = np.zeros((len(X1), len(X1)))
    l1.fill(1/len(X1))
    I2 = np.identity(len(X2))
    l2 = np.zeros((len(X2), len(X2)))
    l2.fill(1/len(X2))
    N1 = np.dot(np.dot(K1, (I1-l1)), K1.T)
    N2 = np.dot(np.dot(K2, (I2-l2)), K2.T)
    N = N1 + N2
    #print(N.shape)
    return N

def classifier(A, X, X1, X2, classes):

    classified = {}
    for i in range(2):
        classified[i] = []
    proj = []
    for i in range(len(X)):
        y = 0
        for j in range(len(X)):
            y = y + A[i]*kernel_function(X[j], X[i])
        proj.append(y)

    theta = np.mean(np.array(proj))
    mis_classification = 0
    #print(proj)
    #print((X[:100] == X1).sum())
    for i in range(len(X)):
        y = proj[i]
        #print("y = ",y)
        if(y <= theta):
            if X[i] in X2:
                #print((X[i] == X2).sum())
                mis_classification = mis_classification + 1
            classified[0].append(X[i])
        else:
            if X[i] in X1:
                mis_classification = mis_classification + 1
            classified[1].append(X[i])
    print("Classification Accuracy", (200-mis_classification)/200)
    print("No. of mis classification", mis_classification)

    return theta

def get_1D_rep(A, X):
    proj = []
    for i in range(len(X)):
        y = 0
        for j in range(len(X)):
            y = y + A[i]*kernel_function(X[i], X[j])
        proj.append(y)
    proj = np.array(proj).reshape(-1,1)
    return proj

def get_plots(A, X, classes, P, theta):
    colors = ['c', 'o']
    plt.plot(theta, np.zeros_like(theta) + 0, 'x', color='b', label='theta')
    plt.legend()
    plt.plot(P[:100], np.zeros_like(P[:100]) + 0, 'o', color='r')
    plt.plot(P[100:], np.zeros_like(P[100:]) + 0, 'o', color='g')
    plt.savefig('Q5d1.png')
    plt.close()

def get_plot2(A, X):
    plt.scatter(X, A, color='b')
    plt.savefig('Q5d2.png')
    plt.close()


def LDA(classes):
    X1 = classes[0]
    X2 = classes[1]
    X = np.append(X1, X2, axis=0)
    M1, M2 = get_M1_M2(X, X1, X2)
    N = get_N(X, X1, X2)
    N = N + 0.001*np.identity(200)
    Alphas = np.dot(np.linalg.inv(N), (M2 - M1))
    OneDRep_X = get_1D_rep(Alphas, X)
    theta = classifier(Alphas, X, X1, X2, classes)
    get_plots(Alphas, X, classes, OneDRep_X, theta)
    get_plot2(Alphas, X)

if __name__ == '__main__':
    c = 2
    n1 = n2 = 100
    xmin = [-3, 2]
    xmax = [-2, 3]
    classes = {}
    X1 = np.load('X1.npy')
    classes[0] = X1
    X2 = np.load('X2.npy')
    classes[1] = X2
    LDA(classes)

    #get_plots(classes)
