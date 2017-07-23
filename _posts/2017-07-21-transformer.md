--- 
layout: post 
title: 有趣的transformer 
date: 2017-07-21 
categories: blog 
tags: [DeepLearning, NLP] 
description: transformer里面的神奇 
--- 

# 有趣的transformer

主要针对google发的两篇文章来讲，一篇是《Attention Is All You Need》，另一篇是《One Model To Learn Them All》。后一篇与今天的主角transformer不太相关，但是使用了transformer来做multi-modal multi-task model。

## Attention Is All You Need

transformer包含两块内容：encoder和decoder

![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic.png)

### 一、multi-head attention

#### scaled dot-product attention

* Q、K、V的shape为：[N, time_step, hidden_size]。
Q和K经过矩阵乘法；
* scale（因为乘完后怕值太大，从而影响从softmax回传的梯度）
* mask（主要是防止句子中当前词右侧的内容对当前词的影响，使用简单的0/1mask，主要用在self-attention上）

具体公式如下：

![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic1.png)
￼
#### multi-head

先将V、K、Q切分，转变为[N, h, time_step, hidden_size/h]或[N*h, time_step, hidden_size/h]的形式，然后再进入scaled dot-product attention处理，得到的结果再concat，之后通过一个linear层。公式如下：
￼
![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic2.png)

![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic3.png)

#### 三种attention

* encoder-decoder attention：使用multi-head attention，输入为encoder的输出和decoder的self-attention输出，其中encoder的self-attention作为key and value，decoder的self-attention作为query
* encoder self-attention：使用multi-head attention，输入的Q、K、V都是一样的（input embedding and positional embedding）
* decoder self-attention：使用masked multi-head attention，输入的Q、K、V都是一样的（output embedding and positional embedding）

#### 使用self-attention的好处

主要考虑计算复杂度、并行性、长依赖路径长度（比如某句话的第一个词和译文最后一个词的信息传递路径）。  
restricted self-attention表示将一句话切分成长度为r的n/r块，再在每块里做attention。

![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic4.png)

### 二、FFN

用了两层Dense层，activation用的都是Relu。可以看成是两层的1*1的1d-convolution。hidden_size变化为：512->2048->512
￼
![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic5.png)

### 三、embedding and softmax

embedding：input embedding、positional embedding和softmax前的linear层共享一个weight matrix，每个embedding layer的输出均乘上sqrt(hidden_size)

### 四、positional embedding

文中说训练的positional embedding与直接使用固定的正弦曲线的效果一致，并且如果直接用固定的位置信息（词的位置从1到time_step），也是一样的。下面主要讲一下使用正弦曲线作为positional embedding。
考虑词序维度(pos)和embedding维度(i)，如果embedding维度位置为偶数，则用sin，否则用cos。pos决定了相位，而i决定了周期/频率。每个位置的embedding值都可以由其余位置线性表出(比如sin(a+b)=sin(a)cos(b)+cos(a)sin(b))。

![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic6.png)
￼
### 五、optimizer
￼
![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic7.png)

### 六、regularization 

用了三个方法：  

* residual dropout

![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic8.png)
￼

* attention dropout

![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic9.png)

* label smoothing
￼
![](http://odjt9j2ec.bkt.clouddn.com/transformer-Pasted%20Graphic10.png)


## One Model To Learn Them All 


MultiModel，实现一个multi modal multi tasks model。

### 一、convolutional blocks

#### dilated convolutional layer

空洞卷积/扩张卷积，目的是为了扩大filter对于图像的respective field（感受到的图像的范围大小）。  
每个红点对应的weight不为0，其余为0。第一个的为1-dilated，就是一般的卷积，感受野为3*3；第二个为2-dilated，它前面接一层1-dilated后，感受野为7*7；第三个为4-dilated，它前面接1-dilated、2-dilated后，感受野为15*15。  

![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic.png)

￼看一维的wavenet中使用的dilated conv：
￼
![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic1.png)

#### depthwise separable convolution 

可以认为就是从将每个channel区分开来，然后分别独立计算。可以参考下图的extreme Inception，它先通过一个1*1的con（可以认为是pointwise的卷积，是对某个点的所有channel加权平均，目的是将输入channel映射到新的channel），再进行depthwise separable convolution。而separable convolution是先depthwise再pointwise的。

￼![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic2.png)


#### 模型结构

convstep用的是separable dilated convolution，然后在hidden2和hidden4中加入residual部分，之后dropout一下。
￼
￼![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic3.png)

![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic4.png)

### 二、attention blocks

#### dot-product attention

使用multi-head dot-product attention mechanism来计算attention。方法和《Attention Is All You Need》当中类似。
￼
![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic5.png)

#### attention

左边是将target data（embedding）和timing embedding直接相加，然后通过两层dilation conv，再做multi-head self-attention；右边则是source data（embedding）分别传入两个pointwise conv，之后同左边输出到一个multi-head attention，得到一个target（target是query）关于source（source是key and value）的attention矩阵。
￼
![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic6.png)

### 三、timing

timing的输出应该和target data的形式一样，都是[batch_size, time_step,  hidden_size]。固定某个sample，对应time step为t，那么对应的hidden部分的第2d维和第2d+1维分别由sin和cos确定，可以认为对hidden部分的奇数位和偶数位分别用sin和cos处理。其中，d决定了正弦曲线的周期，t决定了正弦曲线的相位。

![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic7.png)

### 四、Mixture-of-Experts Blocks 

各种expert blocks的组合。没有详述具体的内容。

### 五、Encoder and Mixer and Decoder

下图的input encoder输出encoded inputs以及I/O Mixer输出encoded outputs，两者作为Decoder的输入。图中白色的左右对称的圆表示concat（猜的）。attention的左侧输入是作为target的，因此左侧的conv层无法访问到未来的信息（具体原因不是很明白，文中没有详述）。下文的long term dependencies可以理解为输入与输出序列之间任意两个元素对应的距离（The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies.）

￼![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic8.png)
￼
￼![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic9.png)

![](http://odjt9j2ec.bkt.clouddn.com/transformer-multimodal-Pasted%20Graphic10.png)

## PDF downloads

[Attention Is All You Need](http://odjt9j2ec.bkt.clouddn.com/transformer-Attention%20Is%20All%20You%20Need.pdf)  
[One Model To Learn Them All](http://odjt9j2ec.bkt.clouddn.com/transformer-One%20Model%20To%20Learn%20Them%20All.pdf)

## References

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)  
[One Model To Learn Them All](https://arxiv.org/pdf/1706.05137.pdf)  
[tensorflow for transformer](https://github.com/Kyubyong/transformer)
