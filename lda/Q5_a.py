import numpy as np
import matplotlib.pyplot as plt

def get_plots(classes):
    colors = ['c', 'o']
    for i, c in zip(classes, colors):
        x = classes[i]
        x = np.array(x)
        #plt.scatter(x, color=c)
        val = 0. # this is the value where you want the data to appear on the y-axis.
        #ar = np.arange(10) # just as an example array
        plt.plot(x, np.zeros_like(x) + val, 'o')
    plt.savefig('Q5a.png')

if __name__ == '__main__':
    c = 2
    n1 = n2 = 100
    xmin = [-3, 2]
    xmax = [-2, 3]
    classes = {}
    X1 = np.random.uniform(low = -1, high = 1, size=(n1,1))
    classes[0] = X1
    X2  = np.random.uniform(low = xmin, high = xmax, size=(n2,1))
    classes[1] = X2
    np.save('X1.npy', X1)
    np.save('X2.npy', X2)

    get_plots(classes)
