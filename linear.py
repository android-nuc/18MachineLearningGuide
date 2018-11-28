import numpy as np
import os
import matplotlib.pyplot as plt
from models import LinearModel
from models import MultinomialModel
from models import StereMultinomial
from models import LWRL


def loadData():
    """
    读取数据
    :return:
    """
    # path = "./data/linear_data.txt"
    path = "./data/temperature.txt"
    file = open(path)
    string = file.read()
    file.close()
    return string


def clipString(string):
    """
    切分字符串  注意dtype
    :param string:
    :return:
    """
    return np.array(string.split("，"), dtype=int)


def drawer(x, y):
    """
    可视化
    :param x:
    :param y:
    :return:
    """
    plt.scatter(x, y)
    plt.show()


def linearModel(X, y):
    model = LinearModel()
    model.train(X, y, 1000, 1e-3)
    model.draw(X, y)


def multinomialModel(X, y):
    model = MultinomialModel()
    model.train(X, y, 5000, 1e-7, 5)
    model.draw(X, y)


def stereMultinomialModel(X, y):
    model = StereMultinomial()
    model.train(X, y, 500000, 20, 1e-5)
    model.draw(X, y)


def lwrl(X, y):
    model = LWRL()
    preds = []
    for i in range(len(X)):
        pred = model.predict(X[i], X, y, k=1)
        preds.append(pred)
    model.draw(X, y, preds)


if __name__ == "__main__":
    raw_data = loadData()
    datalist = clipString(raw_data)
    X = np.arange(start=1, stop=len(datalist) + 1, step=1)
    # drawer(X, datalist)
    # linearModel(X, datalist)
    # multinomialModel(X, datalist)
    stereMultinomialModel(X, datalist)
    # lwrl(X, datalist)
