--- 
layout: post 
title: KMP算法与AC自动机
date: 2017-02-19 
categories: blog 
tags: [算法] 
description: 两种模式匹配的算法
--- 

# KMP算法与AC自动机

KMP算法——用于单模匹配。  
AC自动机——用于多模匹配，需要了解KMP原理和Trie树。

### KMP算法

KMP算法用于单模匹配，比如在一个目标串当中匹配一个模式串。暴力解法就是扫描目标串与模式串，如果发现不匹配，则目标串起始点回溯到原起始点，再后移一位，而模式串回溯到开头0。  
KMP算法能够高效的找出目标串中的匹配串，相比于暴力搜索的O(m*n)的时间复杂度，KMP算法的搜索复杂度为O(m+n)。  

KMP算法的基本原理如下：

举个例子，如果我们的目标串为S，匹配串为P，那么在匹配过程中，于$S_i$处失配：  

$$S_0,S_1,S_2,S_3,...,S_{i-j},S_{i-j+1},...,S_i,S_{i+1},...,S_m\\
~~~~~P_0,P_1,...,P_{j-1},P{j},...\\

$$

$$P_0,$$


### AC自动机

待补充……

### 参考

1. [KMP算法之总结篇](http://www.cnblogs.com/mfryf/archive/2012/08/15/2639565.html)
2. [KMP算法详解](http://blog.csdn.net/yutianzuijin/article/details/11954939/)