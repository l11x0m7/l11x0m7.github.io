--- 
layout: post 
title: RF、GBDT和xgboost原理简述
date: 2016-12-01 
categories: blog 
tags: [机器学习] 
description: 3种树的集成算法比较
--- 

# RF、GBDT和xgboost原理简述

RF：从M个训练样本中随机选取m个样本，从N个特征中随机选取n个特征，然后建立一颗决策树。这样训练出T棵树后，让这k颗树对测试集进行投票产生决策值。RF是一种bagging的思路。可以并行化处理。

GBDT：总共构建T棵树。当构建到第t棵树的时候，需要对前t-1棵树对训练样本分类回归产生的残差进行拟合。每次构建树的方式以及数据集一样，只不过拟合的目标变成了t-1棵树输出的残差。不可并行化处理。

xgboost：总共构建T颗树。当构建到第t颗树的时候，需要对前t-1颗树对训练样本分类回归产生的残差进行拟合。每次拟合产生新的树的时候，遍历所有可能的树，并选择使得目标函数值（cost）最小的树。但是这样在实践中难以实现，因此需要将步骤进行分解，在构造新的树的时候，每次只产生一个分支，并选择最好的那个分支。如果产生分支的目标函数值（cost）比不产生的时候大或者改进效果不明显，那么就放弃产生分支（相当于truncate，截断）。可以并行化处理，效率比GBDT高，效果比GBDT好。

两者都用了CART（分类回归树）的方法。

$$
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable columnalign="right left" rowspacing="3pt" columnspacing="0em" displaystyle="true">
    <mtr>
      <mtd>
        <mi>O</mi>
        <mi>b</mi>
        <msup>
          <mi>j</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mo stretchy="false">(</mo>
            <mi>t</mi>
            <mo stretchy="false">)</mo>
          </mrow>
        </msup>
      </mtd>
      <mtd>
        <mi></mi>
        <mo>&#x2248;<!-- ≈ --></mo>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>n</mi>
        </munderover>
        <mo stretchy="false">[</mo>
        <msub>
          <mi>g</mi>
          <mi>i</mi>
        </msub>
        <msub>
          <mi>w</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>q</mi>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>x</mi>
              <mi>i</mi>
            </msub>
            <mo stretchy="false">)</mo>
          </mrow>
        </msub>
        <mo>+</mo>
        <mfrac>
          <mn>1</mn>
          <mn>2</mn>
        </mfrac>
        <msub>
          <mi>h</mi>
          <mi>i</mi>
        </msub>
        <msubsup>
          <mi>w</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>q</mi>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>x</mi>
              <mi>i</mi>
            </msub>
            <mo stretchy="false">)</mo>
          </mrow>
          <mn>2</mn>
        </msubsup>
        <mo stretchy="false">]</mo>
        <mo>+</mo>
        <mi>&#x03B3;<!-- γ --></mi>
        <mi>T</mi>
        <mo>+</mo>
        <mfrac>
          <mn>1</mn>
          <mn>2</mn>
        </mfrac>
        <mi>&#x03BB;<!-- λ --></mi>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>j</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>T</mi>
        </munderover>
        <msubsup>
          <mi>w</mi>
          <mi>j</mi>
          <mn>2</mn>
        </msubsup>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mi></mi>
        <mo>=</mo>
        <munderover>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>j</mi>
            <mo>=</mo>
            <mn>1</mn>
          </mrow>
          <mi>T</mi>
        </munderover>
        <mo stretchy="false">[</mo>
        <mo stretchy="false">(</mo>
        <munder>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>&#x2208;<!-- ∈ --></mo>
            <msub>
              <mi>I</mi>
              <mi>j</mi>
            </msub>
          </mrow>
        </munder>
        <msub>
          <mi>g</mi>
          <mi>i</mi>
        </msub>
        <mo stretchy="false">)</mo>
        <msub>
          <mi>w</mi>
          <mi>j</mi>
        </msub>
        <mo>+</mo>
        <mfrac>
          <mn>1</mn>
          <mn>2</mn>
        </mfrac>
        <mo stretchy="false">(</mo>
        <munder>
          <mo>&#x2211;<!-- ∑ --></mo>
          <mrow class="MJX-TeXAtom-ORD">
            <mi>i</mi>
            <mo>&#x2208;<!-- ∈ --></mo>
            <msub>
              <mi>I</mi>
              <mi>j</mi>
            </msub>
          </mrow>
        </munder>
        <msub>
          <mi>h</mi>
          <mi>i</mi>
        </msub>
        <mo>+</mo>
        <mi>&#x03BB;<!-- λ --></mi>
        <mo stretchy="false">)</mo>
        <msubsup>
          <mi>w</mi>
          <mi>j</mi>
          <mn>2</mn>
        </msubsup>
        <mo stretchy="false">]</mo>
        <mo>+</mo>
        <mi>&#x03B3;<!-- γ --></mi>
        <mi>T</mi>
      </mtd>
    </mtr>
  </mtable>
</math>
$$

# 参考
[xgboost introduction](https://xgboost.readthedocs.io/en/latest/model.html)

[Practical XGBoost in Python](http://education.parrotprediction.teachable.com/courses/enrolled/practical-xgboost-in-python)