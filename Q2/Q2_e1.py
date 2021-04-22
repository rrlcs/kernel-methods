import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from numpy import linalg


################################################################################

def sample_weight_matrix_W(dim):

    ''' 
    Random sampling of 2 x 1 Weight Matrix
    Input: Dimension
    Output: 2 x 1 W
    '''

    mean = (0,0)
    cov = np.identity(dim)
    return np.random.multivariate_normal(mean, cov, 1)

def sample_bias_vector_b(dim):

    '''
    Randomly Sample Intercept or bias vector
    Input: Dimension
    Output: 2 x 1 b vector
    '''

    mu = 0
    sigma = 1
    return np.random.normal(mu, sigma, (dim,1))

def get_samples_of_xi(n):

    '''
    Sample n 2x1 vectors in range[-3.3]
    Input: n
    Output: 2xn Matrix X
    '''

    X = np.random.uniform(low=-3, high=3, size=(2,n))
    return X

def get_labels_of_xi(X, n):

    '''
    Get Class Labels of each point
    Input: X, n
    Output: n x 1 y vector of class labels
    '''

    y = []
    for i in range(0,n):
        xi = X[:,i].reshape(2,1)
        z = np.dot(W,xi) + b
        sign = np.sign(z)
        y.append(sign)
    y = np.array(y)
    y = y[:, :, 0]
    return y


################################################################################

################################################################################


def kernel_function(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def gram_matrix(X, n_train):
    K = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            K[i,j] = kernel_function(X[:, i], X[:, j])
    return K

def get_P(y, n_train):

    '''
    Returns square matrix P
    Input: Number of Variable to Solve for n_var
    Output: n_var x n_var P matrix
    '''
    K = gram_matrix(X, n_train)
    P = np.outer(y, y) * K
    return P

def get_opt_var(C):

    '''
    Returns Optimized Parameters
    Input: Hyperparameter c
    Output: P, q, G, h
    '''

    #n_var = 103
    n_eqn = 100
    P = matrix(get_P(y, n_train))
    q = matrix(np.ones(n_train) * -1)
    G = matrix(np.vstack((np.diag(np.ones(n_train) * -1), np.identity(n_train))))
    #print(np.array(G).shape)
    h = matrix(np.hstack((np.zeros(n_train), np.ones(n_train) * C)))
    yd = y * (1.0)
    A = matrix(yd.reshape(1, -1))
    b = matrix(0.0)
    return P, q, G, h, A, b

def cvxopt_solver():

    '''
    Call Solver
    Returns solution vector of size n_var x 1
    '''

    P, q, G, h, A, b = get_opt_var(0.1) ## Change Hyperparameter C
    sol = solvers.qp(P, q, G, h, A, b)
    return sol

def get_cvxopt_parameters():

    '''
    Returns W_opt, zeta and b_opt
    '''

    sol = cvxopt_solver()
    alphas = np.ravel(sol['x'])
    return alphas

def get_y_pred(W_opt, b_opt, X, n):
  
    '''
    Returns Predicted Labels
    Input: W_opt, b_opt, X, n
    Output: y_pred
    '''

    y_pred = []
    for i in range(0,n):
        xi = X[:,i].reshape(2,1)
        z = np.dot(W_opt.T,xi) + b_opt
        sign = np.sign(z)
        y_pred.append(sign)
    y_pred = np.array(y_pred)
    return y_pred

def project(X, y, alphas, n):
    y_predict = np.zeros(n)
    K = np.dot(X.T,X)
    alphas = alphas.reshape(100,1)
    temp = np.dot(np.multiply(alphas, y).T, K)
    b = y - temp.T
    ydash = np.dot(np.multiply(alphas, y).T, K)
    y_predict = np.sign(ydash.T + b)
    return y_predict

def get_accuracy(y, y_pred):

    '''
    Returns Accuracy
    '''

    from sklearn import metrics
    return metrics.accuracy_score(y, y_pred)

################################################################################

################################################################################
if __name__ == '__main__':
    
    ''' Set Parameters '''
    n_train = 100
    n_test = 100
    W = sample_weight_matrix_W(2)
    b = sample_bias_vector_b(1)

    ''' Sample Training Data '''
    X = np.load('Data_X_Q2_c.npy')
    y = np.load('Data_y_Q2_c.npy')

    ''' Sample Test Data '''
    X_test = get_samples_of_xi(n_test)
    y_test = get_labels_of_xi(X_test, n_test)

    ''' Training '''
    alphas = get_cvxopt_parameters()
        
    y_pred = project(X, y, alphas, n_train)

    ''' Testing '''
    y_pred_test = project(X_test, y, alphas, n_test)

    ''' Training Accuracy | Test Accuracy | W_opt | b_opt | C '''
    print("Training Accuracy = ", get_accuracy(y, y_pred))
    print("Test Accuracy = ", get_accuracy(y_test, y_pred_test))

