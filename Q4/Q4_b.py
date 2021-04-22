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
            sec = sec + k(y[:,i], y[:,j])
    sec = sec/(d**2)
    third = 0
    for i in range(y.shape[0]):
        third = third + k(x, y[:,i])
    third = 2*(third/d)
    return np.sqrt(first + sec - third)

def get_distance_matrix(E, k):
    d = E.shape[0]
    D = np.zeros((d,d))
    for i in range(d):
        D[i, i] = dist(E[:,i], E, k)
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

