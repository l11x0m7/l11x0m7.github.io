--- 
layout: post 
title: 论文浅见《APPLYING DEEP LEARNING TO ANSWER SELECTION: A STUDY AND AN OPEN TASK》
date: 2016-10-31 
categories: blog 
tags: [NLP, CNN, 论文] 
description: 
--- 

# 论文浅见《APPLYING DEEP LEARNING TO ANSWER SELECTION: A STUDY AND AN OPEN TASK》

## 零、主要目的

建立一个保险领域的QA系统，即客户给出一个问题，在知识库中寻找与之最为匹配的答案。

## 一、 注意点

### 原文1

> a CNN leverages three important ideas that can help improve a machine learning system: sparse interaction, parameter sharing and equivariant representation. Sparse interaction contrasts with traditional neural networks where each output is interactive with each input. In a CNN, the filter size (or kernel size) is usually much smaller than the input size. As a result , the output is only interactive with a narrow window of the input. Parameter sharing refers to reusing the filter parameters in the convolution operations, while the element in the weight matrix of traditional neural network will be used only once to calculate the output. Equivariant representation is related to the idea of k-MaxPooling which is usually combined with a CNN.  

大致内容：  
CNN利用了三个重要的思路，能够帮助改善机器学习系统：稀疏交互(sparse interaction)、参数共享(parameter sharing)和等变表示(equivariant representation)。稀疏交互和传统神经网络的每个输出都与输入进行交互（全连接）形成对比。在CNN中，过滤器尺寸（核大小）通常比输入大小要小很多。因此，CNN网络内部的输出仅仅和一个输入的窄窗交互。而参数共享是指在一次卷积过程中重复使用过滤器的参数（即一个通道channel只使用一个filter），而传统神经网络中的权重矩阵元素只在计算输出的时候被使用一次。等变表示类似于经常与CNN组合在一起的k最大池化的思想（即在一个池中选择k个最大值）。
 
### 原文2

> During training, for each training question Q there is a positive answer A+(the ground truth). A training instance is constructed by pairing this A+ with a negative answer A−(a wrong answer) sampled from the whole answer space.   

大致内容：  
在训练的时候，对每个训练问题Q，总会有一个正答案A+（事实）。一个训练实例是通过从整个答案空间里抽取的这个正答案A+和一个负答案A-（错误答案）形成对。

## 三、细节

1. 两个baseline：BOW and IR model

2. Hinge Loss: $L=max\{0, m-cos(V_Q,V_{A+})+cos(V_Q,V_{A-})\}$，$m$为margin。

### 框架（详细说明下框架2）  
框架1：

![](http://odjt9j2ec.bkt.clouddn.com/qainsurance-1.png)

Q语句和A语句分别处理，各自独立使用HL层、CNN层、P层和T层。

框架2：

![](http://odjt9j2ec.bkt.clouddn.com/qainsurance-2.png)

* 输入Q和A为**[batch_size, sequence_length, embed_size]**
* HL(Hidden Layer)使用tanh函数，输出为**[batch_size, sequence_length, hidden_size]**
* 之后经过CNN层，为带多个filter的单卷积层，输出为**[batch_size, sequence_size-filter_size+1, 1, channels]**，经过P(1-max-pooling)后为**[batch_size, 1, 1, channels]**
* 如果有n个不同类型的filter，则最后输出为**[batch_size, 1, 1, channels*n]**
* 再经过reshape后，可以转为**[batch_size, channels*n]**，之后再计算batch里每个样本的余弦相似度，最后输出为**[batch_size]**。

框架3：

![](http://odjt9j2ec.bkt.clouddn.com/qainsurance-3.png)

框架4：

![](http://odjt9j2ec.bkt.clouddn.com/qainsurance-4.png)

框架5：

![](http://odjt9j2ec.bkt.clouddn.com/qainsurance-5.png)

框架6：

![](http://odjt9j2ec.bkt.clouddn.com/qainsurance-6.png)
