import numpy as np
import dill

def get_kernel_function(filepath):

    '''
    Function for Extracting kernel function
    from the byte code stored in Pickle format

    Input: file path
    Output: Kernel Function reference

    '''

    with open(filepath, 'rb') as in_strm:
        data = dill.load(in_strm)
    kernel_function = dill.loads(data)
    return kernel_function

def dist(x, y, k):
    first = k(x, x)
    sec = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            sec = sec + k(y[i,:], y[j,:])
    sec = sec/(100**2)
    third = 0
    for i in range(y.shape[0]):
        third = third + k(x, y[i,:])
    third = 2*(third/100)
    return np.sqrt(first + sec - third)

def get_distance_matrix(E, k):
    d = E.shape[0]
    D = np.zeros((d,d))
    for i in range(d):
        D[i, i] = dist(E[:,i], E, k)
    return D

def dist_cent(x, cent, k):
    dist = []
    for c in cent:
        dist.append(k(x,x) + k(cent[c],cent[c]) - 200 * k(x,cent[c]))
    return dist

def kernelKMeans(K, d, data, k, max_iter=200):

    # Randomly Initialize K Centroids
    centroids = {}
    for i in range(K):
        centroids[i] = np.random.uniform(size=(1, d))

    for _ in range(1):
        classes = {}
        for i in range(K):
            classes[i] = []
        for i in range(data.shape[0]):
            distances = dist_cent(data[i,:], centroids, k)
            ci = distances.index(min(distances))
            classes[ci].append(data[i,:])
        for c in centroids:
            x = classes[c]
            x = np.array(x)
            if(len(x) != 0):
                centroids[c] = (1/100)*np.mean(x, axis=0)
    return classes, centroids

def get_plot(classes, centroids):
    colors = ['r', 'b', 'g', 'y', 'm']
    for c, i in zip(centroids, colors):
        x = classes[c]
        x = np.array(x)
        if len(x) != 0:
            x = x.reshape(len(x), 2)
            #print(x)
            plt.scatter(x[:, 0], x[:, 1], color = i)
    plt.savefig('Q4c.png')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d = 2
    K = 5
    D = np.load('data.npy')
    k = get_kernel_function('kernel_4a.pkl')
    classes, centroids = kernelKMeans(K, d, D, k)
    get_plot(classes, centroids)

