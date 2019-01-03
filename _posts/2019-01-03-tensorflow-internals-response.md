---
layout: post
title: 《Tensorflow内核剖析》阅读笔记
date: 2019-01-03
categories: blog
tags: [读后感, tensorflow, 笔记]
description: 书籍阅读笔记
---

# 《Tensorflow内核剖析》阅读笔记

这本书主要是讲解了Tensorflow的内核原理，其中包括Tensorflow中的graph构建机制、session构建与运行机制、分布式调度方式等方面。通读下来，个人觉得内容讲解的还比较浅显，里面的代码很多都是作者根据源码简化后的，所以阅读起来比较轻松，不算吃力。对于想要了解tensorflow工作机制以方便在自己构建模型的时候能够更加理解过程的同学，这是一本不错的书，不过如果想要深入理解源码或者优化源码，那么个人觉得该书只能作为一个开始。

下面我将根据章节来划分与梳理自己对于该书的阅读总结。



## References

* [Tensorflow内核剖析](https://github.com/horance-liu/tensorflow-internals)