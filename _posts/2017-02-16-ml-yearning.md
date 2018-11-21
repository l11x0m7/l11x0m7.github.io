--- 
layout: post 
title: 《Machine Learning Yearning》读记
date: 2017-02-16 
categories: blog 
tags: [机器学习] 
description: 吴恩达的新书，讲机器学习（以深度学习为例）在实际场景当中应用时的一些经验。
--- 

# 《Machine Learning Yearning》读记

难得一口气看完一本英文书（主要是篇幅短，哈哈哈）。这本书不是旨在讲授高深的机器学习原理和复杂的推导，而是讲机器学习在实际应用场景当中应用时遇到的一些问题以及解决问题的一些tricks。而在实际应用中，经验和tricks往往比如何实现机器学习算法和应用机器学习理论方法要来得更困难，这需要在长期工作中不断碰壁而积累到的经验。  

本书目前在网上开放了14章。1-4章介绍了机器学习的概况和本书的大致内容。5-12章告知了如何巧妙的使用dev/test数据集。传统的train/test按70%/30%划分，但是在大数据时代，大量的数据并不要求这个启发式的分法。可以适当的减少test集数据比例。在实验中，一定要认清dev和test的作用。一个模型不仅要fit到training set，还要fit到dev和test，并在实际应用场景中保持好的performance。这就要求dev和test的数据同分布，并且能够涵盖大部分的实际应用场景的情况。文中除了介绍dev/test，同时还强调了single-number evaluation metric在评价model时的重要性。如果有多个metrics，则需要挑出一个optimizing metric（即优化目标），其余的作为satisficing metrics（文中举例为应用程序的大小，即硬性条件）。有了上述的思路后，就可以开始进行“Idea”->“Code”->“Experiment”的迭代循环了。  

总的来说，选好dev sets、test sets和metrics；若dev set过拟合，则扩充dev set；若dev/test set不是同分布，则获取同分布的新的dev/test set；如果metric不奏效，则change it。最后，千万不要频繁的在test set上测试并feedback到模型上，这样做会overfit the test set。  

13-14章讲述基本的错误分析方法。从dev set中的misclassified examples（也可以是其他类型的错误例子）选出一部分，然后人工的观察这些错误样本，进行error analysis。至于分析的方法，文中给出了一个简单的trick，就是用spreadsheet来记录每张错误图片的相关特点。这样可以得到包含某个特点的错误图片占错误图片总数的比例，从而按百分比从大到小去解决这些问题。  

这本书还没出完，后续继续关注。后面给出这本书的下载地址，以及此文的翻译笔记（个人觉得还是读英文版比较有意思，英语用词也比较浅显）。再加一个Andrew NG男神在NIPS 2016上的speech。

1. [Machine Learning Yearning book draft(Chap 1-14)](http://bloglxm.oss-cn-beijing.aliyuncs.com/Machine_Learning_Yearning.pdf)
2. [Machine Learning Yearning book draft - 读记](http://mp.weixin.qq.com/s/UCqPBHvre5mn9F2in0nGlQ)
3. [独家 \| 吴恩达NIPS 2016演讲现场直击：如何使用深度学习开发人工智能应用？](http://mp.weixin.qq.com/s/ZbUCh5bi6Ech55qJR2gaxg)