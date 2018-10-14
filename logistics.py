import numpy as np
import matplotlib.pyplot as plt


# 激活函数
def sigmoid(x):
    output = np.array(1 / (1 + np.exp(-x)))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):  # 激活函数的导数
    return output * (1 - output)


dataset = []
try:
    f = open('dataset\\num(lable 1-40).txt', 'r')
    a = f.read()
    data = a.splitlines()
    print("len", len(data))
    for i in data:
        b = i.split(' ')
        b = list(map(float, b))
        dataset.append(b)
finally:
    if f:
        f.close()
dataset = np.array(dataset)
#print(dataset.shape)
#print(type(dataset[0][1]))
# @X input
X = np.array(dataset[:,:-1])
X = X.T
#print(X.shape)
# @y true output row: n,col: 1
y = np.array(dataset[:, -1]).reshape(X.shape[1], 1)
y = y.T  # change to row:1,col: n
y = sigmoid(y).reshape(1,X.shape[1])
print("y  ", y.shape)
print("y[0][0]", y[0][0])
# @w coefficient
w = np.random.rand(X.shape[0]).reshape(X.shape[0],1)
#print(w.shape)
# @b
b = 0
# @r
r = 0.1
J = 0
tp = tn = fp = fn = 0
y_col = []
for i in range(200):
    # Z = w.T * X + b row : 1,col : n
    Z = np.dot(w.T, X) + b
    a = sigmoid(Z)
    tp = tn = fp = fn = 0
    #print(a.shape)
    # @dz y^
    dz = y - a
    #dz = sigmoid_output_to_derivative(dy)
    dw = 1/X.shape[1] * np.dot(X, dz.T)
    db = 1/X.shape[1] * np.sum(dz)
    w = w - r * dw
    b = b - r * db

    for k in range(X.shape[1]):
        if abs(a[0][k] - y[0][k]) <= 0.05:
            if y[0][k] == 0:
                tp += 1
            else:
                fp += 1
        else:
            if y[0][k] == 0:
                tn += 1
            else:
                fn += 1
    acc = (tp + fp) / X.shape[1]
    y_col.append(acc)
    print(i, "   ", acc)

'''
for i in range(10):
    for j in range(X.shape[1]):
        zj = np.dot(w.T, X[j]) + b
        aj = sigmoid(zj)
        cost_j = -(y[0][j] * np.log2(aj) + (1 - y[0][j]) * np.log2(1 - aj))
        J += cost_j
   '''
'''
out = np.dot(w.T, X)+b
tranResult = sigmoid(out)
print("tran"+str(tranResult.shape))
for i in range(X.shape[1]):
    if abs(tranResult[0][i] - y[0][i]) <= 0.05:
        if y[0][i] == 0:
            tp += 1
        else:
            fp += 1
    else:
        if y[0][i] == 0:
            tn += 1
        else:
            fn += 1
acc = (tp+fp)/X.shape[1]
print(acc)
'''
i = np.arange(0,len(y_col),1)
plt.plot(i, y_col, 'r-o')
plt.show()