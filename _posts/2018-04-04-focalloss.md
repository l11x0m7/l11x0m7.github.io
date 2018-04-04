---
layout: post
title: Focal Loss，一个general的idea
date: 2018-04-04
categories: blog
tags: [Deep Learning]
description:
---

# Focal Loss，一个general的idea

相信大家在做数据建模的时候经常会遇见样本不均衡的问题，之前的常规做法有：

* 调整损失函数权重，让样本少的在loss中的权重比例高，反之降低；
* 过采样、欠采样
* OHEM：只保留loss高的样本再训练，忽略简单样本

而虽然上述方法可以让模型关注到样本不平衡问题本身，但是并没有针对这个问题提出比较合理的解释（点一和点二），或者是采用比较复杂（丑陋）的方法（点三）。在Kaiming He的论文Focal Loss for Dense Object Detection中，提出的Focal Loss能够优雅的去尝试解决这个问题。

## 问题本质

* 样本不平衡本身会对模型造成的问题就是让那些极其稀少的样本被淹没在较多的样本中，从而让稀少样本变得不那么重要（对模型来说可以忽略掉）。
* 对于一个问题，大多数样本都是简单易分的，而难分的只占少数。这往往会造成easy problem dominating的问题。大多数的简单样本对loss起主要贡献，占据了主导权，那么那些难分的样本就被模型忽略了。如何让模型能够关注到这些难分样本本身，就是focal loss尝试去解决的问题。

## Focal Loss

对于多分类的cross entropy而言，其损失函数可以写成：

$$
CE(y_t)=-log(\hat{y_t}) \\
\hat{y_t}=softmax(x_t)
$$


上述$y_t$为对应目标的预测概率。如果我们能够让那些loss高的样本拥有较高的权重，而loss较低的拥有较低的权重，那么我们可以在这个loss前面加一个调权公式：

$$
CE(y_t)=-f(y_t)*log(\hat{y_t})
f(y_t)={\alpha}_t*(1-\hat{y_t})^{\gama}
$$

当$\hat{y_t}$趋于1时，那么$f(y_t)$就趋于0，表示这是个easy样本，需要削弱它对loss的贡献；相反，如果$\hat{y_t}$趋于0，那么$f(y_t)$就趋于$\alpha_t$，表示这是个hard样本，需要加强它对loss的贡献。  
这里的两个参数α和γ协调来控制，本文作者采用α=0.25，γ=2效果最好。

## References

* [知乎：如何评价kaiming的Focal Loss for Dense Object Detection？](https://www.zhihu.com/question/63581984)
* [Focal Loss论文阅读笔记](https://blog.csdn.net/qq_34564947/article/details/77200104)
