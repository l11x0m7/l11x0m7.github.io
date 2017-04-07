--- 
layout: post 
title: sigmoid和softmax，cross-entropy和log-likelihood
date: 2017-04-07 
categories: blog 
tags: [DeepLearning] 
description: 有趣的数学现象
--- 

最近在回顾BP的时候，发现了这个有趣的结论：当最后一层使用sigmoid作为激活函数输出并且cost function为cross entropy，或者最后一层使用softmax层并且cost function为negative log likelihood的时候，那么两者传到最后一层的$\delta^L$的形式相等。

现假设：

$$sigmoid:\hat{y}_i=a_i^L=\sigma(z_i^L)$$

$$softmax:\hat{y}_i=a_i^L=\frac{e^{z_i^L}}{\sum_je^{z_j^L}}$$

$$ground truth(one\ hot\ vector):y\ ;\ \ y_k=1\ ;\ \ y_l=0,l\neq{k}$$

$$C_{ce}=-y^T*log(\hat{y})-(1-y^T)*log(1-\hat{y})=-log\hat{y_k}-\sum_{l\neq{k}}log(1-\hat{y_l})$$

$$C_{nll}=-y^T*log(\hat{y})=-\sum_ly_l*log(\hat{y_l})$$

$$\delta^L=\frac{\partial{C}}{\partial{z^L}}\ \ \ \ \ \ \delta_i^L=\frac{\partial{C}}{\partial{z_i^L}}$$

现在我们来推导一下：

#### 对于sigmoid

当$i=k$时（即我们求偏导的那个神经元的输出就是正确类别的那个输出，此时$y_i=y_k=1$）:

$$\frac{\partial{C_{ce}}}{\partial{z_i^L}}=-\frac{\hat{y_i}^{'}}{\hat{y_i}}=-\frac{\sigma^{'}(z_i^L)}{\sigma(z_i^L)}=\hat{y_i}-1$$

同理，当$i\neq{k}$时：

$$\frac{\partial{C_{ce}}}{\partial{z_i^L}}=\frac{\hat{y_i}^{'}}{1-\hat{y_i}}=\frac{\sigma^{'}(z_i^L)}{1-\sigma(z_i^L)}=\hat{y_i}$$

综上：

$$\delta^L=\frac{\partial{C_{ce}}}{\partial{z^L}}=\hat{y}-y$$

#### 对于softmax

当$i=k$时（即我们求偏导的那个神经元的输出就是正确类别的那个输出，此时$y_i=y_k=1$）:

$$\frac{\partial{C_{nll}}}{\partial{z_i^L}}=\frac{\partial{C_{nll}}}{\partial{a_k^L}}\frac{\partial{a_k^L}}{\partial{z_i^L}}=-\frac{1}{a_k^L}*[a_k^L-a_k^L*a_i^L]=a_i^L-1$$

同理，当$i\neq{k}$时：

$$\frac{\partial{C_{nll}}}{\partial{z_i^L}}=\frac{\partial{C_{nll}}}{\partial{a_k^L}}\frac{\partial{a_k^L}}{\partial{z_i^L}}=-\frac{1}{a_k^L}*[-a_k^L*a_i^L]=a_i^L$$

综上：

$$\delta^L=\frac{\partial{C_{nll}}}{\partial{z^L}}=\hat{y}-y$$

### 结论

两者的$\delta^L$在结构上保持一致，但是内部的$\hat{y}$含义却不太一样（虽然都是包含概率信息，但是sigmoid针对的是具体某个神经元，而softmax需要考虑该层所有的神经元）。目前我只发现这两种方式的结构是一样的，如果有读者发现更general的想法，欢迎在底下评论留言。

### 后续

经人指点，原来这个在PRML上有普适性的结论。

![](http://odjt9j2ec.bkt.clouddn.com/math-1.png)