import numpy as np
import matplotlib.pyplot as plt


def get_samples_of_x(n):
    return np.random.uniform(low=-1, high=1, size=(100,1))

def get_samples_of_y(X):
    return np.sin(3*X)

def get_plot(X, Y, y_pred):
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.figure(figsize=(7,5))
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.scatter(X[:], Y[:], label='Data Points')
    plt.scatter(X, y_pred, color='r', label='Line fit')
    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.savefig('Q3d.png')

def fit_model(K, Y, reg):
    
    return np.linalg.inv(K + reg*np.identity(len(Y))).dot(Y)

def get_feature_map(X_k, X_t, k):

    if k==1:
        return np.power(X_t, k)
    else:
        X_k = np.append(X_k, np.power(X_t, k), axis=1)
        return X_k

def predict(X, a, K, Y, reg):
    y_pred = []
    for x in X:
        pred = []
        for xn in X:
            pred.append(kernel_function(xn, x))
        pred = np.array(pred)
        pred = np.dot(pred.reshape(-1, 1).T, a)
        y_pred.append(pred)
    y_pred = np.array(y_pred)[:,:,0]
    return np.dot(K + reg*np.identity(len(Y)), a)

def mse(Y, y_pred):
    return np.sum(np.square(Y-y_pred))

def kernel_function(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def polynomial_kernel(x, y, p=1):				
	return (1 + np.dot(x, y)) ** p

def gram_matrix(X, n_train):
    K = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            #K[i,j] = polynomial_kernel(X[i, 0], X[j, 0])
            K[i,j] = kernel_function(X[i, 0], X[j, 0])
    return K


if __name__ == '__main__':

    n = 100 # Size of Train and Test Data
    reg = 0.001 # Regularizer

    #------------Load Training Data----------#
    
    X = np.load('Data_X_Q3_a.npy')
    Y = np.load('Data_y_Q3_a.npy')

    #-----------Training---------------------#
    
    K = gram_matrix(X, n)
    a = fit_model(K, Y, reg)

    #-----------------Predict----------------#
    
    y_pred = predict(X, a, K, Y, reg)

    #-------------Testing--------------------#

    X_test = get_samples_of_x(n)
    y_test = get_samples_of_y(X_test)
    y_pred_test = predict(X_test, a, K, y_test, reg)

    #-----Mean Squared Error for Train and Test Data-----#
    
    mean_squared_error_train = mse(Y, y_pred)
    mean_squared_error_test = mse(y_test, y_pred_test)

    #----------Display Error-----------------#

    print("Mean Squared Error for Training = ", mean_squared_error_train)
    print("Mean Squared Error for Test = ", mean_squared_error_test)
    get_plot(X, Y, y_pred)
