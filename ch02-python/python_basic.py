# -*- coding: UTF-8 -*-
"""
此脚本用于展示Python的基础语法
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import sys

if sys.version_info[0] == 3:
    from functools import reduce
else:
    pass


# dict基本操作
## 初始化一个dict类型变量x
x = {"a": 1, "b": 2, "c": 3}
print(x)

## 读取x中的一个元素，直接读取或者使用“get”方法
print(x["a"], ",", x.get("a"))
## 读取x中不存在的一个元素，注意直接读取会报错。
try:
    print(x["d"])
except KeyError as e:
    print(repr(e))

## 使用“get”方法返回None值；可以在“get”方法中使用默认值
print(x.get("d"))
print(x.get("d", "No such key"))

## 修改dict
x["c"] = 4
## 插入新的键值对
x["d"] = "5"
print(x)
## 删除键值对
del x["c"]
print(x)


# list基本操作
## 初始化一个list类型变量y
y = ["A", "B", "C", "a", "b", "c"]
print(y)

## 读取y中的元素
print(y[0])
print(y[-1])
print(y[0: 3])
## 查找y中的元素
print(y.index("a"))

## 修改list
y[0] = 4
print(y)
## 在y的最后面插入新元素
y.append(5)
print(y)
## 在指定位置插入新元素
y.insert(3, "new")
print(y)
## 两个list合并，注意和append的区别
print(y + ["d", "e"])
y.append(["d", "e"])
print(y)
## 删除y里面的元素
y.remove("B")
print(y)


# tuple基本操作
## 初始化一个tuple类型变量z
z = ("a", "b", "c", "d", "e")
print(z)

## 读取z中的元素，与list类似
print(z[0])
print(z[-1])
print(z[0: 3])

# lambda表达式和内置函数
## 定义普通函数
def f(a, b):
    return a + b


print(f(1, 2))
## 定义lambda表达式，下面的g和函数f等价
g = lambda x, y: x + y
print(g(1, 2))

## 内置map函数和lambda表达式
## l是一个0到5的列表
l = list(range(6))
print(l)
## 下面的操作将生成一个新的列表，列表里面的元素为l里元素加一
def addOne(data):
    re = []
    for i in data:
        re.append(i + 1)
    return re


print(addOne(l))
## 通过内置的map函数和lambda表达式可以达到同样的效果
# Python2和Python3的map并不兼容，所以使用list(map)
print(list(map(lambda x: x + 1, l)))
## 达到同样功能的列表生成式
print([i + 1 for i in l])

## 计算l中每个元素的两倍和平方，并将两种组成一个列表
## lambda表达式和python函数一样，也可以接受函数作为参数
twoTimes = lambda x: x * 2
square = lambda x: x ** 2
# Python2和Python3的map并不兼容，所以使用list(map)
print([list(map(lambda x: x(i), [twoTimes, square])) for i in l])

## 内置filter函数，选择l中的偶数
# Python2和Python3的filter并不兼容，所以使用list(filter)
print(list(filter(lambda x: x % 2 == 0, l)))

## 内置reduce函数，计算l的和
print(reduce(lambda accumValue, newValue: accumValue + newValue, l, 0))
