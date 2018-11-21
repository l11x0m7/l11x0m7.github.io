--- 
layout: post 
title: 信息论的可视化
date: 2016-10-05 
categories: blog 
tags: [信息论, 可视化] 
description: 信息论的可视化
--- 

# 信息论的可视化

最近看交叉熵和信息散度的时候看到了一篇关于可视化的信息论，感觉挺受启发。原文是英文，想边翻译边学习。篇幅较长，待以后有空补充。  
后来发现已经有人翻译过了，就直接添加到参考了。

主要的收获：

## 交叉熵和KL散度

交叉熵给了我们一种表达两种分布差异的方法。p和q的差异有多大，p关于q的交叉熵就会比p的熵大多少。交叉熵越小，表示分布越接近。

![](http://bloglxm.oss-cn-beijing.aliyuncs.com/visual-info-theory-ce.png)

KL散度的有意思的地方是它就像两个分布之间的距离。衡量着两个分布之间的差异。KL散度越大，两个分布之间差异越大。

![](http://bloglxm.oss-cn-beijing.aliyuncs.com/visual-info-theory-kl.png)

## 参考

1.[Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/)  
2.[可视化信息论](http://blog.csdn.net/xtydtc/article/details/52265952)