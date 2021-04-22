import dill
import numpy as np


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

def get_samples_of_xi(n, knum):
    
    '''
    Sample n 3x1 vectors in range[-5.5]
    Input: n
    Output: 3x100 Matrix X
    '''

    if(knum == 5):
        #pass
        with open('k5sampler.pkl', 'rb') as in_strm:
            data = dill.load(in_strm)
        k5sampler = dill.loads(data)
        X = []
        for i in range(n):
            x = k5sampler()
            X.append(x)
        X = np.array(X)
        X = X[:,:,0].T

    else:
        X = np.random.uniform(low=-5, high=5, size=(3,n))

    return X

def get_gram_matrix(kernel_func, n, knum):

    '''
    Calculates nxn Gram Matrix K
    Input: Kernel Function, n
    Output: Gram Matrix K
    '''

    K = np.zeros((n,n))
    X = get_samples_of_xi(n, knum)
    for i in range(0,n):
        for j in range(0,n):
            x_i = X[:,i].reshape(3,1)
            x_j = X[:,j].reshape(3,1)
            K[i,j] = kernel_func(x_i, x_j)
    return K

def check_symmetric(K, tol=1e-6):

    '''
    Checks if the Matrix is Symmetric or not
    Input = Matrix
    Output: Boolean
    '''

    return np.all(np.abs(K-K.T) < tol)

def is_valid_kernel(K):

    '''
    To check if the Kernel Function is a Valid Kernel or not
    For a Valid Kernel Gram Matrix is Symmetric and Positive Semi-Definite
    Input: Gram Matrix K
    Output: Boolean
    '''
    
    if(check_symmetric(K)):
        return np.all(np.linalg.eigvals(K) >= -1e-6)
    else:
        return False

def check_function1():
    knum = 1
    k1 = get_kernel_function('function1.pkl')
    K = get_gram_matrix(k1, n, knum)
    if(is_valid_kernel(K)):
        print("Kernel Function from function1.pkl is Valid")
    else:
        print("Kernel Function from function1.pkl is Not Valid")


def check_function2():
    knum = 2
    k2 = get_kernel_function('function2.pkl')
    K = get_gram_matrix(k2, n, knum)
    if(is_valid_kernel(K)):
        print("Kernel Function from function2.pkl is Valid")
    else:
        print("Kernel Function from function2.pkl is Not Valid")


def check_function3():
    knum = 3
    k3 = get_kernel_function('function3.pkl')
    K = get_gram_matrix(k3, n, knum)
    if(is_valid_kernel(K)):
        print("Kernel Function from function3.pkl is Valid")
    else:
        print("Kernel Function from function3.pkl is Not Valid")


def check_function4():
    knum = 4
    k4 = get_kernel_function('function4.pkl')
    K = get_gram_matrix(k4, n, knum)
    if(is_valid_kernel(K)):
        print("Kernel Function from function4.pkl is Valid")
    else:
        print("Kernel Function from function4.pkl is Not Valid")


def check_function5():
    knum = 5
    k5 = get_kernel_function('function5.pkl')
    K = get_gram_matrix(k5, n, knum)
    if(is_valid_kernel(K)):
        print("Kernel Function from function5.pkl is Valid")
    else:
        print("Kernel Function from function5.pkl is Not Valid")

## 1 Check for Function1.pkl
if __name__ == '__main__':
    n = 500
    check_function1()
    check_function2()
    check_function3()
    check_function4()
    check_function5()
