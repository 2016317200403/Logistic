import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# 定义sigmoid函数
def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s


def init_param(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    # 前向传播 ：
    P = sigmoid(np.dot(w.T, X) + b)  # 调用前面写的sigmoid函数
    cost = -(np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))) / m

    # 反向传播：
    dZ = P - Y
    dw = (np.dot(X, dZ.T)) / m
    db = (np.sum(dZ)) / m
    return dw, db, cost


def optimize(w, b, X, Y, iter, lr, show_epoch=10):
    # 随机梯度下降法训练模型参数
    costs = []
    for e in range(iter):
        dw, db, cost = propagate(w, b, X, Y)
        w = w - lr * dw
        b = b - lr * db

        if e % show_epoch == 0:
            costs.append(cost)
    return w, b, costs


def predict(w, b, X):
    pred_y = sigmoid(np.dot(w.T, X) + b)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0
    return pred_y


if __name__ == '__main__':
    diabetes = pd.read_csv('diabetes.csv')  # 共768 组数据
    # 划分数据
    x = np.asarray(diabetes.drop('Outcome', 1))
    y = np.asarray(diabetes['Outcome'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=16)
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)
    # x_train = x_train.T
    # x_test = x_test.T
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train - means) / stds
    x_test = (x_test - means) / stds
    x_train = x_train.T
    x_test = x_test.T
    # 初始化w
    w, b = init_param(x_train.shape[0])
    # 设置步长值
    alpha = 0.0001
    num_iters = 30000
    w, b, costs = optimize(w, b, x_train, y_train, num_iters, alpha)
    # 训练后准确度
    predict_train = predict(w, b, x_train)
    predict_test = predict(w, b, x_test)

    acc_train = np.sum(predict_train == y_train) / y_train.shape[1]
    acc_test = np.sum(predict_test == y_test) / y_test.shape[1]
    print(acc_train)
    print(acc_test)
