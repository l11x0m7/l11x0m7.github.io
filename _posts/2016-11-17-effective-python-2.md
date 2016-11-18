--- 
layout: post 
title: 《Effective Python》读书笔记(二) 函数
date: 2016-11-17 
categories: blog 
tags: [Python] 
description: effective python读书笔记2
--- 

# 《Effective Python》读书笔记(二) 函数

### 14.尽量用异常来表示特殊情况，而不返回None

```python
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        raise ValueError('Invalid inputs') from e

x, y = 5, 2
try:
    result = divide(x, y)
except ValueError:
    print('Invalid inputs')
else:
    print('Result is %.1f' % result)
```


### 15.在闭包里使用外围作用域中的变量

变量的搜索顺序：  
1. 当前函数的作用域
2. 任何外围作用域（比如包含当前函数的其他函数）
3. 包含当前代码的那个模块的作用域
4. 内置作用域（比如len与str等函数所在的作用域）

使用nonlocal（python3）和global来定义变量作用域范围：
* nonlocal：仅上升到闭包外那个作用域中的变量
* global：直接到模块作用域的变量

如果是在python2中，可以使用下面方式来代替nonlocal的功能：

```python
# 错误的结果
def test1():
	a = 0
    def test2():
        a += 1
    test2()
    print a
    
>>>
error

def test1():
	a = [0]
    def test2():
        a[0] += 1
    test2()
    print a[0]
    
>>>
1
```

当然，也可以创建辅助类来实现。

### 16.考虑用生成器来改写直接返回列表的函数

使用`yield`来代替return，将函数做成生成器。

```python
def read_visits(data_path):
    with open(data_path) as f:
        for line in f:
            yield int(line)

it = read_visits('my_numbers.txt')
percentages = normalize(it)
print(percentages)
```

### 17.在参数上面迭代时，要小心

由于一个迭代器只能够被使用一次，也就是说从头到尾只能被顺序执行一次。

```python
def read_visits(data_path):
    with open(data_path) as f:
        for line in f:
            yield int(line)

def normalize_copy(numbers):
    numbers = list(numbers)  # 将迭代器展开为列表，可能会非常占用内存
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value / total
        result.append(percent)
    return result

it = read_visits('my_numbers.txt')
percentages = normalize_copy(it)
print percentages

>>>
[11, 26, 61]
```

可以通过传入“产生迭代器的函数”来防止上面的情况。

```python
def normalize_func(get_iter):
    total = sum(get_iter())   # New iterator
    result = []
    for value in get_iter():  # New iterator
        percent = 100 * value / total
        result.append(percent)
    return result

percentages = normalize_func(lambda: read_visits(path))
print percentages 

>>>
[11, 26, 61]
```

当然，也可以使用iter容器（类）。

```python
def normalize(numbers):
    total = sum(numbers)
    result = []
    for value in numbers:
        percent = 100 * value / total
        result.append(percent)
    return result
    
# iter容器
class ReadVisits(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        with open(self.data_path) as f:
            for line in f:
                yield int(line)

visits = ReadVisits(path)
percentages = normalize(visits)
print percentages
```

> 对于`iter()`函数，当visits为iter容器时，`iter(visits)`返回一个新的容器（`iter(visits) is iter(visits)`返回`False`）；当visits为迭代器时，`iter(visits)`返回迭代器本身（`iter(visits) is iter(visits)`返回`True`）

### 18.用数量可变的位置参数减少视觉杂讯

符号*用于函数的位置参数。作为函数的位置形参时实现打包功能，而作为函数的位置实参时实现拆包功能。

### 19.用关键字参数来表达可选的行为

符号\*\*用于函数的关键字参数。

### 20.注意函数的默认参数为动态值的情况

* 参数的默认值，只会在程序加载模块并读到本函数的定义时评估一次，对于{}和[]等动态的值，它只会被定义一次
* 对于以动态值作为实际默认值的关键字参数，应该把形式上的默认值写为None，并在函数的文档字符串里（三个双引号的字符串内容）描述该默认值所对应的实际行为

### 21.尽量用关键字形式来指定参数

在python3中，可以在函数参数中使用*来隔断位置参数和关键字参数。\*前面的位置参数，后面的必须要用关键字参数。

```python
def safe_division_c(number, divisor, *,
                    ignore_overflow=False,
                    ignore_zero_division=False):
    try:
        return number / divisor
    except OverflowError:
        if ignore_overflow:
            return 0
        else:
            raise
    except ZeroDivisionError:
        if ignore_zero_division:
            return float('inf')
        else:
            raise
```

在python2里，可以用\*\*kwargs参数，并用pop方法把期望的关键字参数从kwargs字典取走，如果字典的键里没有那个关键字，那么pop方法的第二个参数就会成为默认值。为了判定调用者是否传入了无效的参数值，则需要通过确认kwargs里是否仍有关键字参数来判定。

```python
def safe_division_d(number, divisor, **kwargs):
    ignore_overflow = kwargs.pop('ignore_overflow', False)
    ignore_zero_div = kwargs.pop('ignore_zero_division', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    try:
        return number / divisor
    except OverflowError:
        if ignore_overflow:
            return 0
        else:
            raise
    except ZeroDivisionError:
        if ignore_zero_div:
            return float('inf')
        else:
            raise

assert safe_division_d(1.0, 10) == 0.1
assert safe_division_d(1.0, 0, ignore_zero_division=True) == float('inf')
assert safe_division_d(1.0, 10**500, ignore_overflow=True) is 0
```

