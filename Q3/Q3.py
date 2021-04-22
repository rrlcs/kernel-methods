import numpy as np
import matplotlib.pyplot as plt

def get_samples_of_x(n):
    return np.random.uniform(low=-1, high=1, size=(100,1))

def get_samples_of_y(X):
    return np.sin(3*X)

def get_plot_a(X, Y):
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.figure(figsize=(7,5))
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.scatter(X[:], Y[:], label='Data Points')
    #plt.plot(X, y_pred, color='r', label='Line fit')
    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.savefig('Q3a.png')

def get_plot_b(X, Y, y_pred):
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.figure(figsize=(7,5))
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.scatter(X[:], Y[:], label='Data Points')
    plt.plot(X, y_pred, color='r', label='Line fit')
    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.savefig('Q3b.png')

def fit_model(X, Y):
    n = np.size(X)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    sdev_cross = np.sum(Y*X - n*mean_X*mean_Y)
    sdev = np.sum(X*X - n*mean_X*mean_X)
    w = sdev_cross / sdev
    b = mean_Y - w*mean_X
    return w, b

def predict(X, w, b):
    return w*X + b

def mse(Y, y_pred):
    return np.sum(np.square(Y-y_pred))

if __name__ == '__main__':
    n = 100
    X = get_samples_of_x(n)
    Y = get_samples_of_y(X)
    np.save('Data_X_Q3_a.npy', X)
    np.save('Data_y_Q3_a.npy', Y)
    w, b = fit_model(X, Y)
    y_pred = predict(X, w, b)
    mean_squared_error = mse(Y, y_pred)
    print("Mean Squared Error = ",mean_squared_error)
    get_plot_a(X, Y)
    get_plot_b(X, Y, y_pred)
    
    
