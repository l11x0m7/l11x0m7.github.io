--- 
layout: post 
title: 《Effective Python》读书笔记(一) 用Pythonic来思考
date: 2016-11-16 
categories: blog 
tags: [Python] 
description: effective python读书笔记1
--- 

# 《Effective Python》读书笔记(一) 用Pythonic来思考

### 1.确认python版本，2或3

### 2.遵循PEP8风格，养成良好编程习惯

### 3.bytes、str和unicode的区别

* 在python3中，bytes是一种包含8位值的序列（utf-8、ascii、gbk等），str是一种包含unicode字符的序列
* 在python2中，str是一种包含8位值的序列（utf-8、ascii、gbk等），unicode是一种包含unicode字符的序列
* 在读写文件时，总以'rb'或'wb'开启文件，表示读写二进制数据


### 4.用辅助函数将复杂的表达式封装起来

### 5.序列切割

### 6.在单次切片时，不要同时指定start、end和stride

可以用itertools.islice，它不允许让start、end和stride为负值，从而避免了负值出现的问题。如果一定要使用三个值，可以先做切割，再做步进。

### 7.用列表推导式来取代map和filter

```python
[x**2 for x in a if x % 2 == 0]
# 等价于
map(lambda x:x**2, filter(lambda x: x%2 == 0, a))
```

当然，字典也可以使用列表推导式，比如字典的反转。

```python
origin_dict = {'a':1, 'b':2}
inverse_dict = {b:a for a, b in origin_dict.iteritems()}
```

### 8.不要使用两个以上表达式的列表推导

### 9.用生成器表达式来改写数据量较大的列表推导

如果用的是小括号()，就会返回列表推导式的迭代器。

```python
# it为迭代器
it = (x for x in [1,2,3,4,5])
print next(it)

>>>
1
```

### 10.尽量用enumerate替代range

### 11.用zip函数同时遍历两个迭代器

如果有两个相互关联的对象，可以直接使用`zip`来遍历。

> 受封装的那些迭代器中，只要有一个耗尽，`zip`就不再产生元组了，此时可以使用`itertools.zip_longest`（python2里有`itertools.izip_longest`）函数

### 12.不要在for和while循环后面写else块

这种else块在整个循环执行完后立刻执行。当碰到循环块里的break而跳出，就不会执行else块了。

### 13.合理利用try/except/else/finally结构中的每个代码块

* 如果try块没有发生异常，就执行else块，否则执行except块（需要被捕捉到才行）
* finally总是执行
* try块处理可能出现异常的内容（比如处理读取的文件内容），except块接收来自try块的可能异常，else块处理接下来的部分，finally块用来清理善后（比如关闭文件）