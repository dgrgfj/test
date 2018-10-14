import copy
import numpy as np
from sklearn import preprocessing


np.random.seed(0)  # 固定随机数生成器的种子，便于得到固定的输出，【译者注：完全是为了方便调试用的]

# compute sigmoid nonlinearity


def sigmoid(x):  # 激活函数
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):  # 激活函数的导数
    return output * (1 - output)

# 数据，二维
dataset = []
label = ["normal","ipsweep","mscan","nmap","portsweep","saint","satan","apache2","back","land","mailbomb",
         "neptune","pod","processtable","smurf","teardrop","udpstorm","buffer_overflow","httptunnel",
         "loadmodule","perl","ps","rootkit","sqlattack","xterm","ftp_write","guess_passwd","imap",
         "multihop","named","phf","sendmail","snmpgetattack","snmpguess","spy","warezclient","warezmaster",
         "worm","xlock","xsnoop"]
le = preprocessing.LabelEncoder()
le.fit(label)
try:
    f = open('dataset\\new.txt', 'r')
    #print(f.read())
    a = f.read()
    data = a.splitlines()
    print("len", len(data))
    for i in data:
        #print(i.split(','))
        b = i.split(',')
        dataset.append(b)
finally:
    if f:
        f.close()
for i in range(len(dataset)):
    dataset[i][-1] = le.transform([str(dataset[i][-1])])[0]

alpha = 0.1  # 学习速率
input_dim = len(dataset[0])  # 每次输入一条数据
hidden_dim = 16  # 隐藏层的神经元节点数，远比理论值要大（译者注：理论上而言，应该一个节点就可以记住有无进位了，但我试了发现4的时候都没法收敛），你可以自己调整这个数，看看调大了是容易更快地收敛还是更慢
output_dim = 1  # 我们的输出是一个数，所以维度为1

# initialize neural network weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1  # 输入层到隐藏层的转化矩阵，维度为41*16， 41是输入维度，16是隐藏层维度
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

# 译者注：np.random.random产生的是[0,1)的随机数，2 * [0, 1) - 1 => [-1, 1)，
# 是为了有正有负更快地收敛，这涉及到如何初始化参数的问题，通常来说都是靠“经验”或者说“启发式规则”，说得直白一点就是“蒙的”！机器学习里面，超参数的选择，大部分都是这种情况，哈哈。。。
# 我自己试了一下用【0, 2)之间的随机数，貌似不能收敛，用[0,1)就可以，呵呵。。。
# 以下三个分别对应三个矩阵的变化
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
# 学习10000个例子
for j in range(len(dataset)):

    overallError = 0  # 每次把总误差清零
    layer_2_deltas = list()  # 存储每个时间点输出层的误差
    layer_1_values = list()  # 存储每个时间点隐藏层的值
    layer_1_values.append(np.zeros(hidden_dim))  # 一开始没有隐藏层，所以里面都是0

    X = dataset[:-1]
    y = dataset[-1] #lable

    # （输入层 + 之前的隐藏层） -> 新的隐藏层，这是体现循环神经网络的最核心的地方！！！
    layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
    # output layer (new binary representation)
    # 隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    # did we miss?... if so, by how much?
    layer_2_error = y - layer_2  # 预测误差是多少
    layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))  # 我们把每一个时间点的误差导数都记录下来
    overallError += np.abs(layer_2_error[0])  # 总误差

    # decode estimate so we can print it out
    d= np.round(layer_2[0][0])  # 记录下预测值

    # store hidden layer so we can use it in the next timestep
    layer_1_values.append(copy.deepcopy(layer_1))  # 记录下隐藏层的值，在下一个时间点用

    future_layer_1_delta = np.zeros(hidden_dim)

    # 前面代码我们完成了所有时间点的正向传播以及计算最后一层的误差，现在我们要做的是反向传播，从最后一个时间点到第一个时间点


    # 我们已经完成了所有的反向传播，可以更新几个转换矩阵了。并把更新矩阵变量清零
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress
    if j % 1000 == 0:
        print("Error:" + str(overallError))
        print("Predict:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")

