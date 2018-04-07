---
layout: post
title: 对GroupNorm的认识
date: 2018-04-07
categories: blog
tags: [DeepLearning, ]
description: 简单描述下Group Norm的思想
---

# 对GroupNorm的认识

之前我已经在normalization上单独做了一章，分别概述了Batch Norm、Layer Norm、Weight Norm以及selu在加速网络收敛上的效果。不过，最先提出这个思路的是BN，之后的则是在其基础上进行改进（或者照瓢画葫芦再来一套？），因此此处我们也是直接将GN的效果有BN直接进行比较（作者在论文也是这么做的）。虽然normalization在NLP任务上并不如CV里常用，但是这种学习的思想还是直接去了解的（万一哪天normalization在某类NLP任务上搞了个大新闻，还是很excited的）。  
对上一节有兴趣的读者可以参考：  
[加速网络收敛——BN、LN、WN与selu](http://skyhigh233.com/blog/2017/07/21/norm/)

## GroupNorm的原理

## 小实验