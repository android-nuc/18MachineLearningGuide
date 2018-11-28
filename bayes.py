import numpy as np
import os


def loadData():
    """
    加载数据
    :return:
    """

    origin_path = "./data"
    raw_strs = []
    lists = []
    pathes = ["sentence.txt", "sentencelabel.txt", "鲍骞月语录.txt"]  # 使用循环加载
    for path in pathes:
        temp = os.path.join(origin_path, path)  # os.path.join是路径连接的函数
        file = open(temp)
        raw_strs.append(file.read())
        file.close()

    list_temp = np.array(raw_strs[0].split("\n"))  # 分割将读取来的字符串转换成list方便用于后面的训练
    lists.append(list_temp)
    list_temp = np.array(raw_strs[1].split(";"), dtype=int)
    lists.append(list_temp)
    list_temp = np.array(raw_strs[2].split("\n"))
    lists.append(list_temp)
    return lists


def get_all_words_vec(train_list):
    """
    得到所有单个文字的向量，参考ppt22页最下面
    :param train_list:
    :return:
    """
    word_set = set()
    for string in train_list:
        for word in string:
            word_set.add(word)
    return np.array(list(word_set))


def string_2_vec(string_list, words_vec):
    """
    将一句话转换成 文字向量的对应的01形式，参考ppt的23页中的每一个01向量
    :param string_list:
    :param words_vec:
    :return:
    """
    words_vec = np.array(words_vec)
    vec_list = []
    for string in string_list:
        vec = np.zeros(len(words_vec))
        for word in string:
            one_hot = (word == words_vec).astype(int)
            vec += one_hot
        vec_list.append(vec)
    print(vec_list)
    return vec_list


def train(train_vec, label):
    """
    学习是1   不学习是0
    目的：计算到p(c) p(w|c) 参考PPT23页
    为什么我这儿有p0_vec和p1_vec？ 因为我是训练了两个模型，一个预测学习，一个预测玩
    p0_vec 和 p1_vec 都是p(w|c)
    pStudy 是 p(c)
    :param train_vec:
    :param label:
    :return:
    """
    train_vec = np.array(train_vec)
    label = np.array(label)
    # p0_vec = np.zeros(len(train_vec[0]))
    # p1_vec = np.zeros(len(train_vec[0]))

    p0_vec = np.zeros(len(train_vec[0]))
    p1_vec = np.zeros(len(train_vec[0]))

    p0_denominator = np.zeros(len(p0_vec))
    p1_denominator = np.zeros(len(p1_vec))
    for index in range(len(train_vec)):
        if label[index] == 0:
            p0_vec += train_vec[index]
            p0_denominator += 1
        if label[index] == 1:
            p1_vec += train_vec[index]
            p1_denominator += 1

    p0_vec = (p0_vec / p0_denominator)
    p1_vec = (p1_vec / p1_denominator)
    pStudy = np.sum(label) / len(label)
    return p0_vec, p1_vec, pStudy


def predict(x_vec, p0_vec, p1_vec, pstudy):
    """
    回到贝叶斯方程，我们现在有了p(w|c) p(c)，还差p(w)
    p(w)是由我们需要预测的句子得来的，参考ppt自己推一下就明白了。

    通过贝叶斯直接计算得到两个分类的概率，最后做比较
    :param x_vec:
    :param p0_vec:
    :param p1_vec:
    :param pstudy:
    :return:
    """
    # p0 = np.sum(x_vec / p0_vec * (1 - pstudy))  #有些书上这里是乘法
    # p1 = np.sum(x_vec / p1_vec * (pstudy))
    for index in range(len(x_vec)):
        if x_vec[index] == 0:
            x_vec[index] = 10 ** 5
    p0 = np.sum(p0_vec / (x_vec)) * (1 - pstudy)
    p1 = np.sum(p1_vec / (x_vec)) * (pstudy)
    if p0 > p1:
        print("玩")
    elif p0 < p1:
        print("学习")
    else:
        print("玩与学习对等")


if __name__ == "__main__":
    # print("hello")
    data_list = loadData()
    # print(data_list)
    train_list = data_list[0]
    labels = data_list[1]
    test_list = data_list[2]

    wordsVec = get_all_words_vec(train_list)
    print(train_list, "\n", wordsVec)
    train_vec = string_2_vec(train_list, wordsVec)

    p0_vec, p1_vec, pstudy = train(train_vec, labels)
    # print(p0_vec, p1_vec, pstudy)
    test_vec = string_2_vec(test_list, wordsVec)
    for item in test_vec:
        predict(item, p0_vec, p1_vec, pstudy)
