--- 
layout: post 
title: RF、GBDT和xgboost原理简述
date: 2016-12-01 
categories: blog 
tags: [机器学习] 
description: 3种树的集成算法比较
--- 

# RF、GBDT和xgboost原理简述

## 简述
 
* RF：从M个训练样本中随机选取m个样本，从N个特征中随机选取n个特征，然后建立一颗决策树。这样训练出T棵树后，让这k颗树对测试集进行投票产生决策值。RF是一种bagging的思路。可以并行化处理。

* GBDT：总共构建T棵树。当构建到第t棵树的时候，需要对前t-1棵树对训练样本分类回归产生的残差进行拟合。每次构建树的方式以及数据集一样，只不过拟合的目标变成了t-1棵树输出的残差。不可并行化处理。

* xgboost：总共构建T颗树。当构建到第t颗树的时候，需要对前t-1颗树对训练样本分类回归产生的残差进行拟合。每次拟合产生新的树的时候，遍历所有可能的树，并选择使得目标函数值（cost）最小的树。但是这样在实践中难以实现，因此需要将步骤进行分解，在构造新的树的时候，每次只产生一个分支，并选择最好的那个分支。如果产生分支的目标函数值（cost）比不产生的时候大或者改进效果不明显，那么就放弃产生分支（相当于truncate，截断）。可以并行化处理，效率比GBDT高，效果比GBDT好。

RF、GBDT和xgboost都用了CART（分类回归树）的方法。

## xgboost原理

### xgboost的模型——Tree Ensemble

Tree Ensemble模型是一系列的分类回归树(CART)。分类的话就是离散值判定，回归的话就是将连续值分段判定（比如age<15）。  
每个叶子节点对应score，每个样本在每棵树里的得分相加，就是这个样本的总得分：

$$\hat{y}_i = \sum_{k=1}^K f_k(x_i), f_k \in \mathcal{F}$$

K表示有K棵树，$f_k$相当于第k棵树，而$F$空间表示整个CART树空间。因此我们的目标函数可以写成：

$$\text{obj}(\theta) = \sum_i^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

$$\begin{split}Obj^{(t)} &\approx \sum_{i=1}^n [g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2\\
&= \sum^T_{j=1} [(\sum_{i\in I_j} g_i) w_j + \frac{1}{2} (\sum_{i\in I_j} h_i + \lambda) w_j^2 ] + \gamma T
\end{split}$$

在模型上，RF用的也是Tree Ensemble，不过它们在训练方式上有所差异。  
RF和提升树的训练差异可以从我们上面的第一部分说的原理里能知道。

### xgboost的训练算法——Tree Boosting

假设我们的目标函数如下：

$$\begin{split}\text{obj} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{i=1}^t\Omega(f_i) \\
\end{split}$$

这个目标函数里，我们的训练目标是$f_t(x_i)$，即对应一棵树。通过递增训练（Additive Training）的方式，我们可以一棵树一棵树的求解。

$$\begin{split}\hat{y}_i^{(0)} &= 0\\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i)\\
\hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i)\\
&\dots\\
\hat{y}_i^{(t)} &= \sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i)
\end{split}$$

那么我们如何从上面的公式中求解最优的T棵树呢？首先考虑第t棵树的目标函数：

$$\begin{split}\text{obj}^{(t)} & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{i=1}^t\Omega(f_i) \\
          & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + constant
\end{split}$$

$l$表示损失函数，$\Omega(f_t)$表示第t棵树的正则化项。损失函数$l$可以是MSE（最小平方误差），也可以用logistic损失函数，也可以同交叉熵损失函数等。那么我们假设已知损失函数，对$l$进行泰勒级数展开到二阶导数，可以得到如下目标函数：

$$\text{obj}^{(t)} = \sum_{i=1}^n [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t) + constant$$

其中，

$$\begin{split}g_i &= \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})\\
h_i &= \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)})
\end{split}$$

去除掉所有常数项后，得到第t棵树的目标函数为：

$$obj^(t)=\sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$

这样，目标函数只依赖于损失函数的一阶导数和二阶导数了。

再考虑正则项，正则项如何定义？考虑树的复杂度，我们可以得到正则项：

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

其中，$\gamma$和$\lambda$是人工设定参数，T为树的叶子节点个数，且

$$w_{q(x)} = f_t(x), w \in R^T, q:R^d\rightarrow \{1,2,\cdots,T\} .$$

表示样本x在第t棵树的得分score。q为样本映射的第t棵树叶子节点的函数（每个叶子节点有一个score）。

这样，我们最终的目标函数为：

$$\begin{split}Obj^{(t)} &\approx \sum_{i=1}^n [g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2\\
&= \sum^T_{j=1} [(\sum_{i\in I_j} g_i) w_j + \frac{1}{2} (\sum_{i\in I_j} h_i + \lambda) w_j^2 ] + \gamma T
\end{split}$$

其中，

$$I_j = \{i|q(x_i)=j\}$$

$$G_j = \sum_{i\in I_j} g_i$$

$$H_j = \sum_{i\in I_j} h_i$$

整合后的目标函数为：

$$\text{obj}^{(t)} = \sum^T_{j=1} [G_jw_j + \frac{1}{2} (H_j+\lambda) w_j^2] +\gamma T$$

这样我们可以根据目标函数求解第t棵树的最优树和最优目标值。

$$\begin{split}w_j^\ast = -\frac{G_j}{H_j+\lambda}\\
\text{obj}^\ast = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda} + \gamma T
\end{split}$$

#### 如何学习具体的树结构

由于我们知道了如何去评价一棵树到底有多好（上面的目标函数），那么我们就可以将构造树的步骤进行分解，每一次只优化一层树。考虑一个节点，我们要将该节点分成两个叶子节点，那么我们获得的分数（此处用gain表示，就是分之前和分之后的目标函数差）

$$Gain = \frac{1}{2} \left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

这个公式中包含：

* 左叶子节点得分
* 右叶子节点得分
* 原叶子节点得分
* 额外的叶子的正则化项

如果gain比$\gamma$稍小，那么我们最好不要增加这个分支。这就是树模型里的剪枝思想。  
通过这种方式，我们不断构造各种分法的树，从而求解得到最佳的树。

# 参考
[xgboost introduction](https://xgboost.readthedocs.io/en/latest/model.html)  
[Practical XGBoost in Python](http://education.parrotprediction.teachable.com/courses/enrolled/practical-xgboost-in-python)