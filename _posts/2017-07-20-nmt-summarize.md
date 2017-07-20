--- 
layout: post 
title: 神经机器翻译综述
date: 2017-07-20 
categories: blog 
tags: [综述，NLP] 
description: 关于神经机器翻译的综述
--- 

# 神经机器翻译综述（到2017年6月）

概要目录，内容不是很详细。需要深入挖掘，可以搜索对应模块关键词内容。


1. 网络架构设计
    * 神经图灵机
    2. 利用记忆机制(attention)提高神经机器翻译
    3. 隐变量神经机器翻译模型
    4. Linear Associative Unit（LAU）：支持深层网络
    5. Facebook: Convolutional Seq2Seq
    6. Google: Transformer
2. 受限的词汇量
    * 未登录词替换（UNK or OOV）
    2. sampled softmax
    3. 基于字母（字）的模型（字母与词语的混合模型，高频词用词向量，生僻词用字母/字向量）
    4. Byte Pair Encoding（BPE）：合并高频字串来切分子词
    5. 使用相似词替代未登录词
3. 先验约束（如何加入先验知识）
    * 基于覆盖率的NMT：如何防止一个词被翻译多次
    2. 一致性训练：正向和反向翻译模型具有互补性（同一对翻译的中翻英的attention矩阵与英翻中的attention矩阵应该互补）
    3. 基于后验正则化加入离散特征（加入规则约束）
4. 训练准则
    * 最小风险准则
    2. 柱搜索
5. 低资源语言翻译
    * 伪数据：翻译单语语料库（利用已有模型一个词一个词翻译），构造伪平行语料库
    2. 半监督学习：同时利用平行语料库和单语语料库（A翻译到B，B再翻译到A，模型共2个部分，目标为减小reconstruct结果误差）
    3. 对偶模型：上同，也是相互翻译，利用DL
    4. 多任务学习：一次翻译多个语言（通过大资源语料库来辅助训练低资源语料库）
    5. 通用语言翻译器（一个模型，多个翻译对，一种zero-shot的思想）
    6. 学生-教师模型：比如A翻译B，A翻译C，B翻译C，那么可以通过A翻译B的模型和B翻译C的模型来辅导A翻译C的模型，使得A->C的模型结果与A->B->C的模型结果KL距离变小
6. 多模态
    * 语音->文本
    2. 文本+图像->文本生成
    3. 视频->描述文本
7. 可视化

> exposure bias：it occurs when a model is only exposed to the training data distribution, instead of its own predictions.比如翻译的decoder的输入是前一个的true label，而不是前一个的prediction