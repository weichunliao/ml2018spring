import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def read_data(fname):
    X,y = [], []
    with open(fname,'r') as f:
        for line in f:
            d = line.split()
            X.append([float(a) for a in d[1:]])
            y.append(int(float(d[0])))
    return np.array(X), np.array(y)

train_X, train_y = read_data('./features.train')
test_X, test_y = read_data('./features.test')

def digit2label(ys, label):
    label_y = []
    for y in ys:
        if y == label:
            label_y.append(+1)
        else:
            label_y.append(-1)
    return np.array(label_y)

def count_err(y, y_hat):
    ndata = len(y)
    err = 0.0
    for i in range(ndata):
        if y[i] != y_hat[i]:
            err += 1
    return (err/ndata)

y12 = digit2label(train_y, 8)
test_y12 = digit2label(test_y, 8)
Cs12 = [10**x for x in[-5, -3, -1, 1, 3]]

clf_list = []
for C in Cs12:
    clf = SVC(C=C, kernel = 'poly', degree = 2, gamma = 1.0, coef0 = 1.0)
    clf_list.append(clf.fit(train_X, y12))

n_sv_list = []
for clf in clf_list:
    n_sv_list.append(sum(clf.n_support_))

# print(n_sv_list)
plt.figure(figsize=(6,4), dpi=120)
plt.title("Q13")
plt.plot(n_sv_list)
plt.xticks(range(len(Cs12)), Cs12)
plt.xlabel('C') 
plt.ylabel('# of support vector')
plt.show()
for c, sv in zip(Cs12, n_sv_list):
    print('C='+str(c)+': '+str(sv))




