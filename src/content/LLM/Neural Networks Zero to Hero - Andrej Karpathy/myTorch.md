---
title: "从零实现myTorch（building micrograd）"
id: "2026-04-07"
date: "2026-04-07"
description: "尝试从零开始复现PyTorch,理解自动求导的过程"
tags: ["PyTorch", "自动求导"]
---

这个想法其实很早就有了，但一直因为毕业设计和比赛没能真正开始。最近系统看完了 Andrej Karpathy 本讲的公开课程和仓库资料，收获很大，于是决定亲手复现一版。本篇文章会记录我从零实现这个项目的思路与过程（[项目地址](https://github.com/AAAAAZBX/myTorch)）。

## 项目介绍

myTorch 是一个模仿 PyTorch 实现的深度学习库，主要用于实现autograd engine (自动梯度引擎)，用来实现**反向传播**。

为了更贴近 PyTorch 的效果，本项目在原项目 micrograd 构建方式上有所区别。

反向传播（英语：Backpropagation，意为误差反向传播，缩写为BP）是对多层人工神经网络进行梯度下降的算法，也就是用链式法则以网络每层的权重为变量计算损失函数的梯度，以更新权重来最小化损失函数。

## 构建 myTorch

### 定义数据结构 Tensor

这里先实现最小版本的 `Tensor`，只保存标量数值并支持基础运算。

```python
class Tensor:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        out = Tensor(self.data + other.data)
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        return out

a = Tensor(1.0)
b = Tensor(2.0)
c = Tensor(3.0)

d = a * b + c # (a.__mul__(b)).__add__(c)

print(d) # Tensor(data=6.0)
```

现在在这个表达式的基础上添加联系组织，用于保留有关哪些值产生了哪些其他值的指针。

```python
class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')
        return out
```

这样便可以知道每个值的子项，并且能够追溯到是哪个运算操作生成了这个值。

为了更加直观地可视化整个计算图，要在类中加入 label 来标识各个变量名，使用下面的代码.

```python
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
```



## 附录:

https://github.com/karpathy/micrograd

[Youtube课程原视频](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

[b站翻译版本](https://www.bilibili.com/video/BV1mqrTBvEaf/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=758fb7e86b317eb62ce96b2962bc1d3a)