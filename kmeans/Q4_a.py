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
    #print(x,y)
    #print("k x x",k(x,x))
    #print("k y y",k(y,y))
    #print("k x y",k(x,y))
    return k(x, x) + k(y, y) - 2 * k(x, y)

def get_distance_matrix(E, k):
    d = E.shape[0]
    D = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            D[i, j] = dist(E[:,i], E[:, j], k)
            #print("D i j = ", D[i, j])
    return D


if __name__ == '__main__':
    d = 10
    #D = np.load('data.npy')
    E = np.identity(d)
    #x = E[:,0].reshape(-1,1)
    #y = E[:,1].reshape(-1,1)
    k = get_kernel_function('kernel_4a.pkl')
    D = get_distance_matrix(E, k)
    print("Sum of Distances = ",np.sum(D))
    #d = dist(x, y, k)
    #print("x", x)
    #print("y", y)
    #print(E.shape)
    #print(x.shape)
    #print(y.shape)
