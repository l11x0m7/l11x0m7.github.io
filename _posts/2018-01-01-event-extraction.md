---
layout: post
title: Survey on Event Extraction
date: 2018-01-01
categories: blog
tags: [事件抽取, NLP]
description: 调研事件抽取
---

# Survey on Event Extraction

## Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks

### 数据集

这篇文章的使用了ACE2005的数据集。

### 问题建模

将任务分为trigger classification 和 argument classification。其中，trigger classification是对每个候选词构造上下文特征和当前词特征（即输入一个词窗的子句）进行分类，判断其所属的事件类型。而对于argument classification则是将句子标为一个trigger与一个argument（如果一句话里有多个argument，则标出来单独作为一个样本），之后将整个句子输入模型。两者均是用的文中提出的dynamic multi-pooling convnet来做，可以看成是一个关注到某个词上的句子分类问题。

### 实验结果

不论是trigger identification还是trigger classification效果都要比规则来的好，同理argument identification和argument role也是一样。

## Leveraging FrameNet to Improve Automatic Event Detection

### 数据集

ACE2005和FrameNet

### 问题建模

这篇论文主要用FrameNet的数据（较多）去补充训练ACE2005（较少）的数据，模型用的比较简单，是一个三层的ANN网络。文中尝试将FrameNet数据转变成ACE的标注方式从而扩充数据集。文中仅尝试了ED（event detection）问题，像event role等问题没有涉及。

### 实验结果

文中提出的从FN中检测事件的效果较好，且基于ACE数据训练的模型在FN中的效果也较好，FN+ACE数据训练能达到更好的效果。

## Modeling Skip-Grams for Event Detection with Convolutional Neural Networks

### 数据集

ACE2005

### 问题建模

本文主要考虑事件检测（event detection，ED）问题以及领域迁移问题（domain adaptation，DA）考虑到传统的卷积操作是只考虑连续的k-gram，而没有考虑非连续的k-gram也会影响到结果。因此本文提出了一种考虑非连续位置的卷积方式（即如果词窗长度n为2*15+1，k为3，那么需要$C_{31}^{3}$种组合的结果，并且在这么多种结果中取出最大的结果（max-pooling）。这样做的话复杂度为O(n^k)，因此需要采用动态规划DP将复杂度降低到了O(nk)。本文利用non-consecutive conv让结果达到了state-of-the-art。并且使用该模型测试在DA上的效果。

### 实验结果

在ED和DA上均达到state-of-the-art的效果。