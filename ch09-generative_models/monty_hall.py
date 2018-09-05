# -*- coding: UTF-8 -*-
"""
此脚本用于展示蒙提霍尔问题的答案
"""


import sys

import numpy as np
import matplotlib.pyplot as plt


def audience_make_choice(n):
    """
    生成奖品所在位置以及观众最初的选择
    """
    prize = np.random.randint(3, size=n)
    init_choice = np.random.randint(3, size=n)
    return prize, init_choice


def host_open_door(prize, init_choice):
    """
    根据奖品所在位置以及观众最初的选择，
    生成主持人开门的门号以及观众更改（如果更改）后的选择
    """
    host_door = []
    for i in range(len(prize)):
        # Python2和Python3的range并不兼容，所以使用list(range)
        doors = list(range(3))
        doors.remove(prize[i])
        if init_choice[i] in doors:
            doors.remove(init_choice[i])
        else:
            pass
        np.random.shuffle(doors)
        # 事实上将下面的语句换成hostDoor.append(min(doors))
        # 或者hostDoor.append(max(doors))，得到的结果一样。
        # 也就是说，不管支持人的选择门的具体策略是什么，得到的
        # 结果都是一样的。
        host_door.append(doors[0])
    host_door = np.array(host_door)
    change_choice = [3] * len(prize) - host_door - init_choice
    return host_door, change_choice


def evaluate(prize, init_choice, change_choice):
    """
    计算两种情况下的获奖概率
    """
    init_win = sum(prize == init_choice)
    change_win = sum(prize == change_choice)
    return init_win, change_win


def visualize(times, init_win, change_win):
    """
    将结果可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["font.size"] = 15
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(times, init_win, label="%s" % "坚持最初的选择")
        ax.plot(times, change_win, "r-.", label="%s" % "更换选择")
        ax.set_xlabel("%s" % "模拟次数")
        ax.set_ylabel("%s" % "概率")
    else:
        ax.plot(times, init_win, label="%s" % "坚持最初的选择".decode("utf-8"))
        ax.plot(times, change_win, "r-.", label="%s" % "更换选择".decode("utf-8"))
        ax.set_xlabel("%s" % "模拟次数".decode("utf-8"))
        ax.set_ylabel("%s" % "概率".decode("utf-8"))
    ax.set_yticks(np.array(range(11)) / 10.0)
    ax.set_xlim([0, max(times)])
    ax.grid(linestyle='--', linewidth=1, axis="y")
    plt.legend(loc="best", shadow=True)
    plt.show()


def simulate():
    """
    通过模拟，解决蒙提霍尔问题
    """
    times = []
    init_win = []
    change_win = []
    for i in np.arange(10, 2000, 20):
        times.append(i)
        prize, init_choice = audience_make_choice(i)
        host_door, change_choice = host_open_door(prize, init_choice)
        _init_win, _change_win = evaluate(prize, init_choice, change_choice)
        init_win.append(_init_win / float(i))
        change_win.append(_change_win / float(i))
    visualize(times, init_win, change_win)


if __name__ == "__main__":
    np.random.seed(1001)
    simulate()
