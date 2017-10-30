# -*- coding: UTF-8 -*-
"""
此脚本用于展示蒙提霍尔问题的答案
"""


import sys

import numpy as np
import matplotlib.pyplot as plt


def audienceMakeChoice(n):
    """
    生成奖品所在位置以及观众最初的选择
    """
    prize = np.random.randint(3, size=n)
    initChoice = np.random.randint(3, size=n)
    return prize, initChoice


def hostOpenDoor(prize, initChoice):
    """
    根据奖品所在位置以及观众最初的选择，
    生成主持人开门的门号以及观众更改（如果更改）后的选择
    """
    hostDoor = []
    for i in range(len(prize)):
        doors = range(3)
        doors.remove(prize[i])
        if initChoice[i] in doors:
            doors.remove(initChoice[i])
        else:
            pass
        np.random.shuffle(doors)
        # 事实上将下面的语句换成hostDoor.append(min(doors))
        # 或者hostDoor.append(max(doors))，得到的结果一样。
        # 也就是说，不管支持人的选择门的具体策略是什么，得到的
        # 结果都是一样的。
        hostDoor.append(doors[0])
    hostDoor = np.array(hostDoor)
    changeChoice = [3] * len(prize) - hostDoor - initChoice
    return hostDoor, changeChoice


def evaluate(prize, initChoice, changeChoice):
    """
    计算两种情况下的获奖概率
    """
    initWin = sum(prize == initChoice)
    changeWin = sum(prize == changeChoice)
    return initWin, changeWin


def visualize(times, initWin, changeWin):
    """
    将结果可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"]=["SimHei"]
    plt.rcParams["font.size"] = 15
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(times, initWin, label="%s" % "坚持最初的选择")
        ax.plot(times, changeWin, "r-.", label="%s" % "更换选择")
        ax.set_xlabel("%s" % "模拟次数")
    else:
        ax.plot(times, initWin, label="%s" % "坚持最初的选择".decode("utf-8"))
        ax.plot(times, changeWin, "r-.", label="%s" % "更换选择".decode("utf-8"))
        ax.set_xlabel("%s" % "模拟次数".decode("utf-8"))
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
    initWin = []
    changeWin = []
    for i in np.arange(10, 2000, 20):
        times.append(i)
        prize, initChoice = audienceMakeChoice(i)
        hostDoor, changeChoice = hostOpenDoor(prize, initChoice)
        _initWin, _changeWin = evaluate(prize, initChoice, changeChoice)
        initWin.append(_initWin/float(i))
        changeWin.append(_changeWin/float(i))
    visualize(times, initWin, changeWin)


if __name__ == "__main__":
    np.random.seed(1001)
    simulate()
