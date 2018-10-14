import numpy as np
import matplotlib.pyplot as plt


# 激活函数
def sigmoid(x):
    # output = np.array(1 / (1 + np.exp(-x)))
    output = (abs(x) + x) / 2
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(x):  # 激活函数的导数
    # print("output ", output.shape)
    output = np.where((abs(x) + x) / 2 < 0, 0, 1)
    return output
    # return output * (1 - output)

def print_parameters(W):
    W = W.reshape(1,W.shape[0]*W.shape[1])
    for i in range(5):
        print(W[0][i], end="  ")
    print()

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
# print(dataset.shape)
# print(type(dataset[0][1]))
# @X input
X = np.array(dataset[:, :-1])
X = X.T
print(X.shape)
# @y true output row: n,col: 1
y = np.array(dataset[:, -1]).reshape(X.shape[1], 1)
y = y.T  # change to row:1,col: n
y = y.reshape(1, X.shape[1])
print("y  ", y.shape)
print("y[0][0]", y[0][0])

r = 0.001       # 学习率
hidden_dim = 16   # 隐藏层节点

np.random.seed(1)
w_1 = np.random.random((hidden_dim, X.shape[0]))
b_1 = np.zeros([1, X.shape[1]])
w_2 = np.random.random((1, hidden_dim))
b_2 = np.zeros([1, X.shape[1]])
# print_parameters(w_1)
# print_parameters(w_2)
cost = 0
y_col = []
cost_ = []
# 迭代10次
for i in range(200):
    tp = tn = fp = fn = 0
    Z_1 = np.dot(w_1, X) + b_1
    a_1 = sigmoid(Z_1)
    Z_2 = np.dot(w_2, a_1) + b_2
    a_2 = sigmoid(Z_2)
    dZ_2 = a_2 - y
    # print("dz_2 ", dZ_2.shape)
    cost = 1 / dZ_2.shape[1] * np.sum(np.dot(dZ_2, dZ_2.T))
    cost_.append(cost)
    # print(a_2)
    # print(w_1)
    # print(b_1)
    # print(w_2)
    # print(b_2)
    # print(b_1)
    dw_2 = 1/X.shape[1] * np.dot(dZ_2, a_1.T)
    db_2 = 1/X.shape[1] * np.sum(dZ_2, axis=1, keepdims=True)
    dZ_1 = np.dot(w_2.T, dZ_2) * sigmoid_output_to_derivative(Z_1)
    dw_1 = 1/X.shape[1] * np.dot(dZ_1, X.T)
    db_1 = 1/X.shape[1] * np.sum(dZ_1, axis=1, keepdims=True)

    w_1 = w_1 - r * dw_1
    b_1 = b_1 - r * db_1
    w_2 = w_2 - r * dw_2
    b_2 = b_2 - r * db_2
    # print_parameters(w_1)
    # print_parameters(w_2)
    for k in range(X.shape[1]):
        if abs(a_2[0][k] - y[0][k]) < 0.5:
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
    # print( "prama  ", dw_1,db_1,dw_2,db_2)

i = np.arange(0, len(y_col), 1)
j = np.arange(0, len(cost_), 1)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(i, y_col)
plt.subplot(2, 1, 2)
plt.plot(j, cost_)
plt.show()
