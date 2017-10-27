# -*- coding: UTF-8 -*-


def wordCount(data):
    """
    输入一个字符串列表，统计列表中字符出现的次数

    参数
    ----
    data : list[str], 需要统计的字符串列表

    返回
    ----
    re : dict, 结果hash表，key为字符串，value为对应的出现次数
    """
    re = {}
    for i in data:
        re[i] = re.get(i, 0) + 1
    return re


if __name__ == "__main__":
    data = ["ab", "cd", "ab", "d", "d"]
    print "The result is %s" % wordCount(data)
    # The result is {'ab': 2, 'd': 2, 'cd': 1}
