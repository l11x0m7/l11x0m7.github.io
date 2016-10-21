--- 
layout: post 
title: LDA主题模型之python模块
date: 2016-10-21 
categories: blog 
tags: [NLP, LDA] 
description: LDA模型的python模块
--- 

# LDA模型的python模块

首先需要安装lda，`pip install lda`。


```python
# -*- encoding:utf-8 -*-
import lda
import numpy as np
import lda.datasets

# 载入文档-词矩阵
# reuters为路透社新闻
X = lda.datasets.load_reuters()
print("type(X): {}".format(type(X)))
print("shape: {}\n".format(X.shape))
print(X[:5, :5])

# 输出：
'''
type(X): <type 'numpy.ndarray'>
shape: (395L, 4258L)

[[ 1  0  1  0  0]
 [ 7  0  2  0  0]
 [ 0  0  0  1 10]
 [ 6  0  1  0  0]
 [ 0  0  0  2 14]]
'''
```

X有395行，4258列。说明有395个文档，4258个词汇。

```python
# 词汇
vocab = lda.datasets.load_reuters_vocab()
print("type(vocab): {}".format(type(vocab)))
print("len(vocab): {}\n".format(len(vocab)))
print(vocab[:6])


# 输出
'''
type(vocab): <type 'tuple'>
len(vocab): 4258

('church', 'pope', 'years', 'people', 'mother', 'last')
'''
```

```python
# 标题/文档
titles = lda.datasets.load_reuters_titles()
print("type(titles): {}".format(type(titles)))
print("len(titles): {}\n".format(len(titles)))
print(titles[:2])  # 前两篇文章的标题

# 输出
'''
type(titles): <type 'tuple'>
len(titles): 395

('0 UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20', '1 GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21')
'''
```

```python
# 训练模型，20个话题，迭代1000次，默认取alpha=0.1，eta（或者beta）=0.01。
model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)
model.fit(X)
```

```python
# 每个topic的词矩阵，每行表示一个话题的词概率，和为1
topic_word = model.topic_word_
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))

# 输出:
'''
type(topic_word): <type 'numpy.ndarray'>
shape: (20L, 4258L)
'''
```

```python
# 显示每个话题的topn的词
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(-topic_dist)][:n]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
    
# 输出：
'''
*Topic 0
- church conversion neighbouring statements outrage
*Topic 1
- deliver voted ladies flight 960
*Topic 2
- deliver protesters complex taught grandmother
*Topic 3
- church stuff cdu ss performances
*Topic 4
- deliver golf honorary hill hailed
*Topic 5
- deliver conversion neighbouring statements outrage
*Topic 6
- church kim fast walk recall
*Topic 7
- deliver hermannsburg zealand philip lasted
*Topic 8
- deliver hailed filled kim enjoyed
*Topic 9
- church golf honorary hill hailed
*Topic 10
- church listed surrounded golf honorary
*Topic 11
- church enjoyed fast walk recall
*Topic 12
- church performances listed surrounded honorary
*Topic 13
- jailed managed practice obviously chanted
*Topic 14
- church breaking rebuild milos quality
*Topic 15
- church midway tribune protesters complex
*Topic 16
- church bubis croatian leon celebrates
*Topic 17
- deliver protesters complex grandmother neighbour
*Topic 18
- church protesters complex taught grandmother
*Topic 19
- church 1998 worried citizens prigione
'''
```
```python
# 每个文档的topic矩阵，每行代表一个文档的话题分布，和为1
doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))

# 输出：
'''
type(doc_topic): <type 'numpy.ndarray'>
shape: (395, 20)
'''
```

```python
# 显示前10个文档的top1的话题
for n in range(10):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}".format(n, topic_most_pr))


# 输出：
'''
doc: 0 topic: 8
doc: 1 topic: 1
doc: 2 topic: 14
doc: 3 topic: 8
doc: 4 topic: 14
doc: 5 topic: 14
doc: 6 topic: 14
doc: 7 topic: 14
doc: 8 topic: 14
doc: 9 topic: 8
'''
```