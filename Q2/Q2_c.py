import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers

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
    Get True Class Labels of each point
    Input: X, n
    Output: n x 1 y vector of class labels
    '''

    y = []
    for i in range(0,n):
        xi = X[:,i].reshape(2,1)
        xi1 = xi[0,0]
        xi2 = xi[1,0]
        z = xi1**2 + (xi2**2)/2
        if(z <= 2):
            sign = -1
        else:
            sign = 1
        y.append(sign)
    y = np.array(y).reshape(n,1)
    #print(y.shape)
    #y = y[:, :, 0]
    return y

def get_plot(X, y):
    
    '''
    Plots the Data Points and Label them with their\
    corresponding Class labels.
    Input: X, y
    Output: 2-D plot
    '''
    y = y.reshape(y.shape[0])
    font = {'family' : 'normal',
        'size'   : 10}
    plt.rc('font', **font)
    plt.figure(figsize=(7,5))
    plt.xlabel('X [0, :]')
    plt.ylabel('X [1, :]')
    plt.text(1, 2.6, 'n=100 points sampled', bbox=dict(facecolor='red', alpha=0.5))
    plt.scatter(X[0, :], X[1, :], c=y, cmap='winter')
    plt.savefig('Q2c.png')

############################################ Functions for Question 2 a Ends Here! ###############################################

############################################ Functions for Question 2 b Starts Here! ###############################################

def get_P(n_var):

    '''
    Returns square matrix P
    Input: Number of Variable to Solve for n_var
    Output: n_var x n_var P matrix
    '''

    P = np.zeros((n_var,n_var))
    P[0,0], P[1,1] = 1.0, 1.0
    return P

def get_q(n_var, c):

    '''
    Returns vector q
    Input: Number of Variables n_var and Hyperparameter c
    Output: n_var x 1 q vector
    '''

    q = np.full((n_var, 1), c)
    q[0,0], q[1,0], q[n_var-1,0] = 0.0,0.0,0.0
    return q

def get_G(X, y, n_var, n_eqn):

    '''
    Returns Coefficient Matrix for Linear Constraints
    Intput: Data Points X, Labels y, Number of Variables n_var, Number of Constraints n_eqn
    '''

    neg_y = -y
    zeta = np.full((n_eqn, n_eqn), -1.0)
    b = np.full((1, n_eqn), 1.0)
    X = np.vstack((X, b[0,:]))
    X = X.T
    X_new = []
    for i in range(n_eqn):
        X_new.append(X[i,:] * neg_y[i,0])
    X_new = np.array(X_new)
    G_1 = X_new[:,[0,1]].T
    b_coef = X_new[:,2].reshape(n_eqn,1).T
    for i in range(n_eqn):
        G_1 = np.vstack((G_1, zeta[i,:]))
    G_1 = np.vstack((G_1, b_coef[0,:]))
    G_2 = np.zeros((n_var, n_eqn))
    for i in range(n_eqn):
        G_2[i+2, i] = -1.0
    G_final = np.append(G_1, G_2, axis=1)
    return G_final.T

def get_h(n_eqn):

    '''
    Returns h vector
    Input: Number of Constraints
    Output: n_eqn x 1 vector
    '''

    h1 = np.full((n_eqn, 1), -1.0)
    h2 = np.full((n_eqn, 1), 0.0)
    h = np.vstack((h1, h2))
    return h

def get_opt_var(c):

    '''
    Returns Optimized Parameters
    Input: Hyperparameter c
    Output: P, q, G, h
    '''

    n_var = 103
    n_eqn = 100
    P = matrix(get_P(n_var))
    q = matrix(get_q(n_var, c))
    G = matrix(get_G(X, y, n_var, n_eqn))
    h = matrix(get_h(n_eqn))
    return P, q, G, h

def cvxopt_solver():

    '''
    Call Solver
    Returns solution vector of size n_var x 1
    '''

    P, q, G, h = get_opt_var(0.99) ## Change Hyperparameter C
    sol = solvers.qp(P, q, G, h)
    return sol

def get_cvxopt_parameters():

    '''
    Returns W_opt, zeta and b_opt
    '''

    sol = cvxopt_solver()
    W_opt = sol['x'][0:2]
    zeta = sol['x'][2:102]
    b_opt = sol['x'][102]
    return W_opt, zeta, b_opt

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
    y_pred = y_pred[:, :, 0]

    return y_pred

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
    X = get_samples_of_xi(n_train)
    y = get_labels_of_xi(X, n_train)
    
    np.save('Data_X_Q2_c.npy', X)
    np.save('Data_y_Q2_c.npy', y)

    ''' Sample Test Data '''
    X_test = get_samples_of_xi(n_test)
    y_test = get_labels_of_xi(X_test, n_test)

    ''' Training '''
    W_opt, zeta, b_opt = get_cvxopt_parameters()
    y_pred = get_y_pred(W_opt, b_opt, X, n_train)

    ''' Testing '''
    y_pred_test = get_y_pred(W_opt, b_opt, X_test, n_test)
    
    ''' Get Plot for Question 2.a '''
    get_plot(X, y)

    ''' Training Accuracy | Test Accuracy | W_opt | b_opt | C '''
    print("Training Accuracy = ", get_accuracy(y, y_pred))
    print("Test Accuracy = ", get_accuracy(y_test, y_pred_test))
    print("W_opt = ", W_opt)
    print("b_opt = ", b_opt)
    print("C = ", 0.99)

