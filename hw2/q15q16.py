import numpy as np

def read_data(fname):
    X,y = [], []
    with open(fname,'r') as f:
        for line in f:
            d = line.split()
            X.append([1.0]+[float(a) for a in d[:-1]])
            y.append(int(d[-1]))
    return np.array(X), np.array(y)

def linear_kernel(x1, x2):
    return np.dot(x1,x2)

def set_kernel_matrix(X, kernel):
    size = X.shape[0]
    K = np.zeros((size,size))
    for i in range(size):
        for j in range(i+1):
            K[i,j] = kernel(X[i],X[j])
            K[j,i] = K[i,j]
    return K

def solve_beta(K_mat, lamda, y):
    return np.linalg.solve(K_mat + np.eye(K_mat.shape[0]) * lamda, y)

def pred(test_x, X, beta, kernel):
    score = np.array([kernel(test_x, v) for v in X])
    score = score* beta
    score = np.sum(score)
    return int(np.sign(score))

def bootstrap_resample(X, y , resample_n):
    n_data = len(X)
    resample_i = np.floor(np.random.rand(resample_n)*n_data).astype(int)
    X_resample = X[resample_i]
    y_resmaple = y[resample_i]
    return X_resample, y_resmaple


if __name__ == '__main__':

    X, Y = read_data('/home/chun/Desktop/ml2018spring/hw2/hw2_lssvm_all.dat')
    X_train = X[:400] 
    Y_train = Y[:400]
    X_test = X[400:]
    Y_test = Y[400:]

    print('#Q15')
    print('#Q16')

    for lamda in (0.01, 0.1, 1, 10, 100):
        ein_predictions = [0]*len(Y_train)
        eout_predictions = [0]*len(Y_test)
        for _ in range(250):
            resample_X, resmaple_y = bootstrap_resample(X_train, Y_train, 400)
            K = set_kernel_matrix(resample_X, linear_kernel)
            beta = solve_beta(K, lamda, resmaple_y)
            ein = 0
            for i in range(len(Y_train)):
                x = X_train[i]
                y = Y_train[i]
                ein_predictions[i] += pred(x, resample_X, beta, linear_kernel)
            eout = 0
            for i in range(len(Y_test)):
                x = X_test[i]
                y = Y_test[i]
                eout_predictions[i] += pred(x, resample_X, beta, linear_kernel)
        Ein = np.sum(np.sign(ein_predictions) != Y_train)/len(Y_train)
        Eout = np.sum(np.sign(eout_predictions) != Y_test)/len(Y_test)
        print('lambda:', lamda,', Ein:', Ein, ', Eout:', Eout)
