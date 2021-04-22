import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import dill

def dist(x, y, k):
    return k(x, x) + k(y, y) - 2 * k(x, y)

def get_distance_matrix(E, mu_k, k):
    d = E.shape[0]
    D = np.zeros((d,d))
    for i in range(d):
        mu_k = np.sum(k(E[:,i], np.ones(d))) / d
        D[i, i] = dist(E[:,i], mu_k, k)
            #print("D i j = ", D[i, j])
    return D

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

kernel = get_kernel_function('kernel_4a.pkl')

X = np.load('data.npy')
m=X.shape[0] #number of training examples
n=X.shape[1] #number of features. Here n=2
n_iter=100
K=2 # number of clusters
Centroids=np.array([]).reshape(n,0)

for i in range(K):
    rand=rd.randint(0,m-1)
    Centroids=np.c_[Centroids,X[rand]]

Output={}

EuclidianDistance=np.array([]).reshape(m,0)
for k in range(K):
    print(Centroids[:,k])
    #print(X)
    tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
    #tempDist = get_distance_matrix(X, Centroids[:,k])
    #print(tempDist)
    EuclidianDistance=np.c_[EuclidianDistance,tempDist]
C=np.argmin(EuclidianDistance,axis=1)+1

Y={}
for k in range(K):
    Y[k+1]=np.array([]).reshape(2,0)
for i in range(m):
    Y[C[i]]=np.c_[Y[C[i]],X[i]]

for k in range(K):
    Y[k+1]=Y[k+1].T

for k in range(K):
    Centroids[:,k]=np.mean(Y[k+1],axis=0)

for i in range(n_iter):
     #step 2.a
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(K):
        tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      
    C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b
    Y={}
    for k in range(K):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X[i]]

    for k in range(K):
        Y[k+1]=Y[k+1].T

    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=0)
    Output=Y

plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.title('Plot of data points')
plt.savefig('save1.png')

color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.savefig('save2.png')
