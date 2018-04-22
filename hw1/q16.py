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

# for q16
gamma16 = [10**x for x in [-1, 0, 1, 2, 3]]
y16 = digit2label(train_y, 0)

np.random.seed(5566)
data_idx = np.arange(len(train_X))
min_Eval_idx_list = []
for _ in range(100):
    np.random.shuffle(data_idx)
    val_idx = data_idx[:1000]
    new_train_idx = data_idx[1000:]
    val_X = train_X[val_idx]
    new_X = train_X[new_train_idx]
    val_y = y16[val_idx]
    new_y = y16[new_train_idx]
    tmp_err_list = []
    for g in gamma16:
        clf = SVC(kernel='rbf', C = 0.1, gamma = g)
        clf.fit(new_X, new_y)
        pred = clf.predict(val_X)
        err = count_err(val_y, pred)
        tmp_err_list.append(err)
    min_Eval_idx_list.append(tmp_err_list.index(min(tmp_err_list)))
    
# print(min_Eval_idx_list)
times=[]
for i in range(5):
    times.append(min_Eval_idx_list.count(i))
# print(times)


plt.figure(figsize=(8,5), dpi=100)
plt.title("Q16")
plt.bar(left=np.arange(-1,4,1), height=times, width=1,align="center",yerr=0.000001)
# plt.hist(min_Eval_idx_list, bins=np.arange(-1,3,1))
# plt.xticks(np.arange(len(gamma16)), gamma16)
plt.xlabel("log10(gamma)")
plt.ylabel("the number of selected times")
# plt.xlim(-1, 5)
plt.ylim(0,80)
plt.show()
for g, e in zip(gamma16, times):
    print('gamma='+str(g)+': '+str(e))

