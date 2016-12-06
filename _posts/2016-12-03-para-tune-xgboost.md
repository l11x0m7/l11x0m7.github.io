--- 
layout: post 
title: xgboost完全调参指南
date: 2016-12-03 
categories: blog 
tags: [机器学习] 
description: 翻译xgboost调参指南
--- 

# xgboost调参指南

## 介绍

如果在使用你的预测模型的时候事情没有进展的那么顺利，可以考虑使用XGboost。XGBoost算法是很多数据科学家强有力的武器。它是一个高度复杂的算法，强大到足以解决任何非规则数据。  
使用XGBoost建模很简单，但是使用XGBoost来改进模型是很困难的。这个算法使用了很多参数。为了改进模型，调参非常必要。需要调节哪些参数，以及如何获取最理想的参数值，这些问题都很难回答。  
这篇文章最适合那些XGBoost使用新手。在这篇文章里，我们可以学习到调参的艺术，以及一些关于XGBoost方面的有用的信息。当然，我们会使用数据集来在Python中实践。

## 内容表

接下来按照以下三个大块来讲：

1. The XGBoost Advantage
2. Understanding XGBoost Parameters
3. Tuning Parameters (with Example)

### 1.The XGBoost Advantage

我总是向往这个算法在预测模型里的提升能力。当我在探索它的表现和高准确率下的科学原理的时候，我发现了它的很多优点：  

##### 正则化

* 标准的梯度提升机器（GBM）的实现没有像XGBoost那样的正则项，因此它能够在过拟合方面有所帮助。
* 实际上，XGBoost也被认为是一种“正则提升”技术。  

##### 并行处理

* XGBoost实现了并行处理，和GBM比起来非常快。
* 但是我们也知道提升是一个序列处理过程，因此如何才能做到并行化？我们知道每一棵树能够只根据之前的那一颗来建立，那么是什么阻碍了我们并行化建树？可以查看这个链接来深入探索：[Parallel Gradient Boosting Decision Trees](http://zhanpengfang.github.io/418home.html)  
* XGBoost可以在Hadoop上实现。

##### 高自由度

* XGBoost允许用户自定义优化目标和评估标准。
* 这增加了模型的一个全新的维度，并且并不会限制我们所能做的东西。

##### 处理缺失值

* XGBoost有内建的方法来处理缺失值。
* 用户只需要提供一个不同值，而不是观察并将其作为一个参数。XGBoost在遇到缺失值的时候总是尝试着寻找不同的方式并学习如何去填充缺失值。  

##### 树剪枝

* 当在分割的过程中遇到负损失时，GBM会停止从一个节点产生分支。因此这更像是一种贪婪算法。
* 而XGBoost先产生分支直到最大深度，之后再开始回溯剪枝，并移除哪些不能够获得正收益的分割。
* 另一个这么做的优点是当我们遇到一个负损失分割的时候，比如-2，那么如果接下来的划分为+10。如果是GBM，则是会在遇到-2的时候停止产生分支。但是XGBoost则会继续产生分支，这会使得最终的总分支得分为+8，从而保留这个分支。  

##### 内建交叉验证

* XGBoost允许用户在每次提升的迭代过程中跑一次交叉验证，因此这很容易在跑一次的过程中得到最优的提升迭代次数。
* 这不像GBM一样跑一个grid search并且只有固定的值能够被测试到。

##### 可以衔接到已存在的模型上

* 用户从它之前一次运行的最后一步迭代中开始训练XGBoost模型。这在某类具体的应用中有非常显著的优势。
* sklearn里的GBM的实现同样有这个特性，因此在这一点上GBM和XGBoost一致。

深入理解：

* [XGBoost Guide – Introduction to Boosted Trees](http://xgboost.readthedocs.io/en/latest/model.html)

## XGBoost Parameters



## 参考

* [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

