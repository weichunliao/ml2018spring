import numpy as np

def read_data(fname):
    X,y = [], []
    with open(fname,'r') as f:
        for line in f:
            d = line.split()
            X.append([float(a) for a in d[:-1]])
            y.append(int(d[-1]))
    return np.array(X), np.array(y)

def rbf_kernel(gamma, x_i, x_j):
    x = x_i - x_j
    k_ij = np.exp(-gamma * np.dot(x,x))
    return k_ij

def set_kernel_matrix(X, kernel):
    size = X.shape[0]
    K = np.zeros((size,size))
    for i in range(size):
        for j in range(i+1):
            K[i,j] = kernel(X[i],X[j])
            K[j,i] = K[i,j]
    return K

def solve_eq(K_mat, lamda, y):
    return np.linalg.solve(K_mat + np.eye(K_mat.shape[0]) * lamda, y)

def pred(test_x, X, beta, kernel):
    score = np.array([kernel(test_x, v) for v in X])
    score = score * beta
    score = np.sum(score)
    return np.sign(score)

if __name__ == '__main__':

    X, Y = read_data('./hw2_lssvm_all.dat')
    X_train = X[:400] 
    Y_train = Y[:400]
    X_test = X[400:]
    Y_test = Y[400:]

    print('#Q11')
    print('#Q12')
    for gamma in [32, 2, 0.125]:
        for lamda in [0.001, 1, 1000]:
            kernel = lambda x1, x2: rbf_kernel(gamma, x1, x2)
            K = set_kernel_matrix(X_train, kernel)
            beta = solve_eq(K, lamda, Y_train)
            ein = 0
            for x, y in zip(X_train, Y_train):
                ein += int(pred(x, X_train, beta, kernel) != y)
            eout = 0
            for x, y in zip(X_test, Y_test):
                eout += int(pred(x, X_train, beta, kernel) != y)
            print('gamma:{}, lambda:{}\n\t Ein:{}\t Eout:{}'.format(gamma, lamda, ein, eout))

