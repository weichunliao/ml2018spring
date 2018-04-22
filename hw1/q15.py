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

gamma = [10**x for x in [0,1,2,3,4]]
y15 = digit2label(train_y, 0)
test_y15 = digit2label(test_y, 0)
Eout_list = []
for g in gamma:
    clf = SVC(kernel='rbf', C = 0.1, gamma = g)
    clf.fit(train_X, y15)
    pred = clf.predict(test_X)
    err = count_err(test_y15, pred)
    Eout_list.append(err)
plt.figure(figsize=(6,4), dpi=120)
plt.title("Q15")  
plt.plot(Eout_list)
plt.xticks(range(len(gamma)), gamma)
plt.xlabel('gamma') 
plt.ylabel('Eout')
plt.show()
for g, e in zip(gamma, Eout_list):
    print('gamma='+str(g)+': '+str(e))
