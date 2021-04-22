import numpy as np
import dill
import matplotlib.pyplot as plt

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
    #print(x,y)
    #print("k x x",k(x,x))
    #print("k y y",k(y,y))
    #print("k x y",k(x,y))
    return k(x, x) + k(y, y) - 2 * k(x, y)

def kernel_KMeans(K, d, data, kernel, max_iter=100):
    centroids = {}
    for i in range(K):
        centroids[i] = np.random.uniform(size=(1, d))
    #print("centr = ",centroids)
    for _ in range(max_iter):
        classes = {}
        distances = []
        for k in range(K):
            classes[i] = []

        for i in range(data.shape[0]):
            for cent in centroids:
                c = centroids[cent]
                #print(c)
                dd = dist(data[i,:], c, kernel)
                #if dd==0:
                    #print(data[i, :], c, dd)
                #print("centr = ",centroids[cent])
                #print("centr = ",centroids[cent])
                distances.append(dd)
            #print('distances',distances)
            #print(data[i, :])
            #print(data[i, :])
            #distances = np.array(distances)
            classes[i] = distances.index(min(distances))
            #print(classes[i])
        for cent in centroids:
            den = 0
            sx = sy = 0
            for i in classes:
                #print("i =",i)
                if i == cent:
                    sx = sx + data[i,0]
                    sy = sy + data[i, 1]
                    den = den + 1
            #print("den",den)
            centroids[cent] = np.array([sx/den, sy/den])
            #print(centroids)
    return centroids, classes

def get_distance_matrix(E, k):
    d = E.shape[0]
    D = np.zeros((d,d))
    for i in range(d):
        mu_k = np.sum(k(E[:,i], np.ones(d))) / d
        D[i, i] = dist(E[:,i], mu_k, k)
            #print("D i j = ", D[i, j])
    return D

def get_plots(centroids, classifications):

    colors = 100*["g","r","c","b","k"]
    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

    for classification in classifications:
        color = colors[classification]
        for featureset in classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    plt.savefig('Q4c.png')

if __name__ == '__main__':
    d = 2
    K = 2
    data = np.load('data.npy')
    #E = np.identity(d)
    #x = D[:,0].reshape(-1,1)
    #y = E[:,1].reshape(-1,1)
    k = get_kernel_function('kernel_4a.pkl')
    centroids, classes = kernel_KMeans(K, d, data, k)
    get_plots(centroids, classes)
    #D = get_distance_matrix(data, k)
    print("centroids = ",centroids)
    print("classes = ",classes)
    #d = dist(x, y, k)
    #print("x", x)
    #print("y", y)
    #print(E.shape)
    #print(x.shape)
    #print(y.shape)
