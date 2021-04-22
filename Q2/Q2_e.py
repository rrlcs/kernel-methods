import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from numpy import linalg


############################################ Functions for Question 2 a Start Here! ###############################################

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


############################################ Functions for Question 2 a Ends Here! ###############################################

############################################ Functions for Question 2 b Starts Here! ###############################################


def kernel_function(x, y, sigma=500.5):
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

    P, q, G, h, A, b = get_opt_var(1000.001) ## Change Hyperparameter C
    sol = solvers.qp(P, q, G, h, A, b)
    return sol

def get_cvxopt_parameters():

    '''
    Returns W_opt, zeta and b_opt
    '''

    sol = cvxopt_solver()
    alphas = np.ravel(sol['x'])

    # Support vectors have non zero lagrange multipliers
    sv = (alphas > 1e-4).flatten()

    ind = np.arange(len(alphas))[sv]
    alphas = alphas[sv]
    sv_X = X[:,sv]
    #print("sv_X",sv_X)
    sv_y = y[sv[:], :]
    #print("sv_y",sv_y)
    #print("%d support vectors out of %d points" % (len(alphas), n_train))
    
    #print("alphas",alphas.shape)
    #print("y*alphas", (y*alphas).shape)
    
    K = gram_matrix(X, n_train)

    # Intercept
    b = 0
    for n in range(len(alphas)):
        b = b + sv_y[n]
        b = b - np.sum(alphas * sv_y * K[ind[n],sv])
    b = b / len(alphas)
    #print("b", b)
    # Weight vector
    alphas = alphas[alphas > 1e-4]
    return b, alphas, sv_X, sv_y

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
    #print(y_pred.shape)
    #y_pred = y_pred[:, :, 0]

    return y_pred

def project(X, alphas, sv_X, sv_y, b_opt, n_train):
    y_predict = np.zeros(n_train)
    for i in range(n_train):
        s = 0
        for k in range(len(alphas)):
            s = s + alphas[k] * sv_y[k, 0] * kernel_function(X[:, i], sv_X[:, k])
        y_predict[i] = s
    return np.sign(y_predict + b_opt)

def get_accuracy(y, y_pred):

    '''
    Returns Accuracy
    '''

    from sklearn import metrics
    return metrics.accuracy_score(y, y_pred)

############################################ Functions for Question 2 b Ends Here! ###############################################

########################################################## Entry Point ###########################################################
if __name__ == '__main__':
    
    ''' Set Parameters '''
    n_train = 100
    n_test = 50
    W = sample_weight_matrix_W(2)
    b = sample_bias_vector_b(1)

    ''' Sample Training Data '''
    X = np.load('Data_X_Q2_c.npy')
    #X = get_feature_map(X)
    y = np.load('Data_y_Q2_c.npy')

    ''' Sample Test Data '''
    X_test = get_samples_of_xi(n_test)
    #X_test = get_feature_map(X_test)
    y_test = get_labels_of_xi(X_test, n_test)

    ''' Training '''
    b_opt, alphas, sv_X, sv_y = get_cvxopt_parameters()
    #print(W_opt)
    #y_pred = get_y_pred(W_opt, b_opt[0], X, n_train).astype(np.int)
    #y_pred = y_pred[:, :, 0]
    
    y_pred = project(X, alphas, sv_X, sv_y, b_opt, n_train)
    #print("y_pred", y_pred.reshape(100,1))

    ''' Testing '''
#    y_pred_test = get_y_pred(W_opt, b_opt, X_test, n_test).astype(np.int)
#    y_pred_test = y_pred_test[:, :, 0]
    
    ''' Get Plot for Question 2.a '''
    #get_plot(X, y)

    ''' Training Accuracy | Test Accuracy | W_opt | b_opt | C '''
    print("Training Accuracy = ", get_accuracy(y, y_pred))
    #print("Test Accuracy = ", get_accuracy(y_test, y_pred_test))
    #print("W_opt = ", W_opt)
    #print("b_opt = ", b_opt)
    #print("alphas = ", alphas)
    #print("C = ", 0.001)

