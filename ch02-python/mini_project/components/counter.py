# -*- coding: UTF-8 -*-
"""
此脚本用于实现计数器
"""


def wordCount(data):
    """
    输入一个列表，统计列表中各个元素出现的次数

    参数
    ----
    data : list, 需要统计的列表

    返回
    ----
    re : dict, 结果hash表，key为原列表中的元素，value为对应的出现次数
    """
    re = {}
    for item in data:
        re[item] = re.get(item, 0) + 1
    return re
