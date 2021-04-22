import numpy as np
import matplotlib.pyplot as plt

def get_plot_k(X, Y, y_pred, k):
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.figure(figsize=(7,5))
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.scatter(X[:], Y[:], label='Data Points')
    plt.scatter(X, y_pred, color='r', label='Line fit')
    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    filename = 'Q3c_' + str(k) +'.png'
    plt.savefig(filename)

def fit_model(X, Y):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    X = np.concatenate((np.ones(shape=X.shape[0]).reshape(-1, 1), X), 1)
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def get_feature_map(X_k, X_t, k):

    if k==1:
        return np.power(X_t, k)
    else:
        X_k = np.append(X_k, np.power(X_t, k), axis=1)
        return X_k

def predict(X_k, model):
    b = model[0]
    W = model[1:]
    y_pred = []
    for row in X_k:
        pred = b
        for xi, wi in zip(row, W):
            pred = pred + (wi*xi)
        y_pred.append(pred)
    return y_pred

def mse(Y, y_pred):
    return np.sum(np.square(Y-y_pred))

if __name__ == '__main__':
    n = 100
    X = np.load('Data_X_Q3_a.npy')
    Y = np.load('Data_y_Q3_a.npy')
    X_k = X
    for k in range(1, 11):
        X_k = get_feature_map(X_k, X, k)
        model = fit_model(X_k, Y)
        y_pred = predict(X_k, model)
        mean_squared_error = mse(Y, y_pred)
        print("k = ", k)
        print("Mean Squared Error = ",mean_squared_error)
        get_plot_k(X, Y, y_pred, k)
    
    
