import numpy as np
from sklearn.svm import SVR

def read_data(fname):
    X,y = [], []
    with open(fname,'r') as f:
        for line in f:
            d = line.split()
            X.append([1.0]+[float(a) for a in d[:-1]])
            y.append(int(d[-1]))
    return np.array(X), np.array(y)

def rbf_kernel(gamma, x_i, x_j):
    x = x_i - x_j
    k_ij = np.exp(-gamma * np.dot(x,x))
    return k_ij

if __name__ == '__main__':

    X, Y = read_data('./hw2_lssvm_all.dat')
    X_train = X[:400] 
    Y_train = Y[:400]
    X_test = X[400:]
    Y_test = Y[400:]

    print('#Q13')
    print('#Q14')

    ndata = len(X_train)
    for gamma in [32,2,0.125]:
        for lamda in [0.01, 0.1, 1, 10, 100]:
            svr = SVR(kernel='rbf', epsilon=0.5, C=lamda, gamma=gamma)
            fit = svr.fit(X_train, Y_train)
            ein = np.sum(np.sign(fit.predict(X_train))!=Y_train)/len(Y_train)
            eout = np.sum(np.sign(fit.predict(X_test))!=Y_test)/len(Y_test)
            print('gamma:{}, lambda:{}\n\t Ein:{}\tEout:{}'.format(gamma, lamda, ein, eout))
            
