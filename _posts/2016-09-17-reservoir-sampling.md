--- 
layout: post 
title: Reservoir sampling
date: 2016-09-17 
categories: blog 
tags: [leetcode, 算法] 
description: 水库采样
--- 

# Reservoir sampling

> Reservoir sampling is a family of randomized algorithms for randomly choosing a sample of k items from a list S containing n items, where n is either a very large or unknown number. Typically n is large enough that the list doesn't fit into main memory.

>——Wiki

存储采样是指从一个包含n个对象的列表S中随机抽取出k个对象作为样本，n要么很大要么未知。典型的n通常大到无法将整个列表存入主内存。

这里，我们主要考虑如何通过这个思路产生我们想要的随机样本。通常，在考虑选取数据的时候，会按照等概选取。我们接下来就针对等概选取来说明。

假设有n个对象，我们要从中等概选取k个对象，步骤如下：

* 先将第一个对象放入内存，即选中该对象
* 对每一个后面的对象<img src="http://chart.googleapis.com/chart?cht=tx&chl=i" style="border:none;">
	* 有<img src="http://chart.googleapis.com/chart?cht=tx&chl=1/i" style="border:none;">的概率会用新值覆盖旧的值
	* 有<img src="http://chart.googleapis.com/chart?cht=tx&chl=1-1/i" style="border:none;">的概率会丢弃新的值
* 根据上面的情况，如果总共有k个对象，那么对象1会以概率<img src="http://chart.googleapis.com/chart?cht=tx&chl=\frac12*\frac23*\ldots\frac{n-1}n=\frac{1}{n}" style="border:none;">被选中，对象2会以概率<img src="http://chart.googleapis.com/chart?cht=tx&chl=1*\frac12*\frac23*\ldots\frac{n-1}n=\frac{1}{n}" style="border:none;">被选中，同理对象3会以概率<img src="http://chart.googleapis.com/chart?cht=tx&chl=\frac13*\ldots\frac{n-1}n=\frac{1}{n}" style="border:none;">被选中

综上，可以看到，对于序列中的n个对象，均会以等概方式被选到。

现在通过几个leetcode的例子来看下。


