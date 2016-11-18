--- 
layout: post 
title: 《Effective Python》读书笔记（四） 内置模块
date: 2016-11-18 
categories: blog 
tags: [Python] 
description: effective python读书笔记4
--- 

# 《Effective Python》读书笔记（四） 内置模块

### 42.用functools.wraps定义函数修饰器

修饰器能够在被修饰的函数执行之前和执行完毕之后，分别运行一些附加代码。。因此可以在修饰器里面访问原函数的参数和返回值。下面举一个fibonacci的递归调用的例子。

```python
def trace(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print '%s(%r, %r)->%r' % (func.__name__, args, kwargs, result)
        return result
    return wrapper

@trace
def fibonacci(n):
    if n in (0, 1):
        return n
    else:
        return (fibonacci(n-2), fibonacci(n-1))

fibonacci(3)

>>>
fibonacci((1,), {})->1
fibonacci((0,), {})->0
fibonacci((1,), {})->1
fibonacci((2,), {})->(0, 1)
fibonacci((3,), {})->(1, (0, 1))
```

上面的修饰相当于是`fibonacci = trace(fibonacci)`，也就是说调用`fibonacci(3)`等价于调用`trace(fibonacci)(3)`等价于调用`wrapper(3)`。而且fibonacci已经不是原来的函数了，而是`wrapper`函数，通过`help`函数可以查看得知：

```python
help(fibonacci)
>>>
Help on function wrapper in module  __main__:

wrapper(*args, **kwargs)
```

而使用functools模块里的wraps辅助函数来修饰wrapper可以解决这个问题:

```python
from functools import wraps
def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print '%s(%r, %r)->%r' % (func.__name__, args, kwargs, result)
        return result
    return wrapper

@trace
def fibonacci(n):
    if n in (0, 1):
        return n
    else:
        return (fibonacci(n-2), fibonacci(n-1))

fibonacci(3)

>>>
# 结果和上面一样
fibonacci((1,), {})->1
fibonacci((0,), {})->0
fibonacci((1,), {})->1
fibonacci((2,), {})->(0, 1)
fibonacci((3,), {})->(1, (0, 1))
```

```python
help(fibonacci)
>>>
Help on function fibonacci in module  __main__:

fibonacci(n)
```

### 43.考虑用contextlib和with语句来改写可复用的try/finally代码

以logging模块为例，来看如何使用with来避免重复写try/finally模块。

* logging模块的默认级别为WARNING，它只会打印严重程度大于等于自身级别的消息
* logging.getLogger()返回的是logging的默认logger，它的默认级别为WARNING
* logging.getLogger(name)则获得名为name的新的logger，默认级别为WARNING
* contextlib里有contextmanager(情境管理器)修饰器，可以让被修饰的函数能够用with来操作
* with后接的函数可以有返回值，该返回值由函数里的yield抛出，并被with后面的as接收

```python
import logging
def log_info():
    logging.debug('some debug data')
    logging.error('error log here')
    logging.debug('more debug data')
log_info()


from contextlib import contextmanager
@contextmanager
def debug_logging(level):
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)

time.sleep(1)
with debug_logging(logging.DEBUG):
    print 'Inside:'
    log_info()
time.sleep(1)
print 'After:'
time.sleep(1)
log_info()

>>>
ERROR:root:error log here
Inside:
DEBUG:root:some debug data
ERROR:root:error log here
DEBUG:root:more debug data
After:
ERROR:root:error log here
```

现在加入一个新的logger，并用as接收。

```python
from contextlib import contextmanager
@contextmanager
def debug_logging(level, name):
    logger = logging.getLogger(name)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)

time.sleep(1)
with debug_logging(logging.DEBUG, 'my_logger') as logger:
    logger.debug('This is my message!')
    logging.debug('This will not print!')

>>>
DEBUG:my_logger:This is my message!
```


### 44.用copyreg(或copy_reg)实现可靠的pickle操作

* pickle能将python对象序列化为字节流，它包含如何构建原始的python对象（比如自己定义的类的结构信息）信息，因此不安全
* json的序列化结构简单，比较安全。陌生人通信一般用json来传递
* python2里用的是copy_reg

使用pickle存储的时候，需要考虑一下三种情况：  

* 新的类中添加了新的属性
* 新的类中删除了原有某些属性
* 新的类名称更改


现在考虑一个游戏状态类，里面存储玩家的等级和血量，并用pickle进行dump和load。

```python
class GameState(object):
    def __init__(self):
        self.level = 0
        self.lives = 4

state = GameState()
state.level += 1  # Player beat a level
state.lives -= 1  # Player had to try again

import pickle
state_path = 'game_state.bin'
with open(state_path, 'wb') as f:
    pickle.dump(state, f)

with open(state_path, 'rb') as f:
    state_after = pickle.load(f)
print(state_after.__dict__)

>>>
{'lives': 3, 'level': 1}
```

#### 为缺失的属性提供默认值

现在我要在GameState类里加入得分的属性，那么在load之前旧的GameState类对象的时候就会出问题。这时候需要用copyreg来防止这种情况。

```python
class GameState(object):
    def __init__(self):
        self.level = 0
        self.lives = 4
        self.points = 0

with open(state_path, 'rb') as f:
    state_after = pickle.load(f)
print(state_after.__dict__)

assert isinstance(state_after, GameState)

>>>
# 输出和上面的例子一样，但是缺少新加入的points属性
{'lives': 3, 'level': 1}
```

为了防止出现上面的问题，我们可以使用copyreg来设定默认值，相当于在unpickle的时候重新生成新的类对象。

```python
# 给初始化参数设定默认值
class GameState(object):
    def __init__(self, level=0, lives=4, points=0):
        self.level = level
        self.lives = lives
        self.points = points

# 用来注册copy_reg的函数,其返回值包含unpickle时调用的函数,以及调用该函数时输入的参数
def pickle_game_state(game_state):
    kwargs = game_state.__dict__
    return unpickle_game_state, (kwargs, )

def unpickle_game_state(kwargs):
    return GameState(**kwargs)

import copy_reg
# 表示注册GameState类的pickle方法为pickle_game_state
copy_reg.pickle(GameState, pickle_game_state)

state = GameState()
state.points += 1000
serialized = pickle.dumps(state)
state_after = pickle.loads(serialized)
print(state_after.__dict__)

# 设置一个新的类,增加magic属性
class GameState(object):
    def __init__(self, level=0, lives=4, points=0, magic=5):
        self.level = level
        self.lives = lives
        self.points = points
        self.magic = magic

state_after = pickle.loads(serialized)
print(state_after.__dict__)

>>>
{'magic': 5, 'points': 1000, 'lives': 4, 'level': 0}
{'magic': 5, 'points': 1000, 'lives': 4, 'level': 0}
```

#### 用版本号来管理类

当在新的类中去掉了一些属性时，则可以考虑版本管理方法。比如去除lives属性，载入旧的GameState类对象。

```python
class GameState(object):
    def __init__(self, level=0, points=0, magic=5):
        self.level = level
        self.points = points
        self.magic = magic


def pickle_game_state(game_state):
    kwargs = game_state.__dict__
    # pickle的时候，会把对象存储为当前最新的版本2
    kwargs['version'] = 2
    return unpickle_game_state, (kwargs,)

def unpickle_game_state(kwargs):
    version = kwargs.pop('version', 1)
    #  相当于判定是否是版本1，如果是，则去除对应属性，从而统一到新版本
    if version == 1:
        kwargs.pop('lives')
    return GameState(**kwargs)

copy_reg.pickle(GameState, pickle_game_state)
state_after = pickle.loads(serialized)
print(state_after.__dict__)

>>>
{'points': 1000, 'magic': 5, 'level': 0}
```

#### 固定的引入路径

当类名称改变后，旧的类对象就不能够正常unpickle了。这时候如果使用copyreg，可以将序列化后的数据和unpickle_game_state绑在一起。

不用copyreg时，pickle数据头的信息是绑定到具体类名称的：

```
c__main__
GameState
```

使用copyreg时，pickle数据头的信息是绑定到unpickle方法的：

```
c__main__
unpickle_game_state
```

给出实例：

```python
# 创建旧的类
state = GameState()
# 序列化旧的类
serialized = pickle.dumps(state)
# 删除旧的类,创建新的类,相当于修改类名称
# 清空一下已经注册的copyreg信息
copy_reg.dispatch_table.clear()
del GameState
class BetterGameState(object):
    def __init__(self, level=0, points=0, magic=5):
        self.level = level
        self.points = points
        self.magic = magic

# 不变
def pickle_game_state(game_state):
    kwargs = game_state.__dict__
    kwargs['version'] = 2
    return unpickle_game_state, (kwargs,)

# 用新的类名称取代旧的
def unpickle_game_state(kwargs):
    version = kwargs.pop('version', 1)
    if version == 1:
        kwargs.pop('lives')
    return BetterGameState(**kwargs)

# 注册
copy_reg.pickle(BetterGameState, pickle_game_state)


print(serialized[:35])
after_state = pickle.loads(serialized)
print(after_state.__dict__)

>>>
c__main__
unpickle_game_state
p0
((

{'points': 0, 'magic': 5, 'level': 0}
```


### 45.应该使用datetime模块处理本地时间，而不是用time模块

协调世界时（UTC）是一种标准的时间描述方式，它与时区无关。

#### time模块

time模块不能够很好的协调处理各个时区。最好用time来转换UTC时间和宿主计算机所在时区的时间。对于其他转换，可以用datetime。

```python
# UTC时间转当地时间
now = 1407694710
local_tuple = time.localtime(now)
time_format = '%Y%m%d %H:%M:%S'
time_str = time.strftime(time_format, local_tuple)
print time_str

>>>
20140811 02:18:30

# 当地时间转UTC时间
time_tuple = time.strptime(time_str, time_format)
utc_now = time.mktime(time_tuple)
print utc_now

>>>
1407694710.0
```


#### datetime模块

datetime也可以完成上述time的功能。

```python
#! python3
from datetime import datetime, timezone

now = datetime(2014, 8, 10, 18, 18, 30)
# 转为utc时间
now_utc = now.replace(tzinfo=timezone.utc)
# 此时的now_utc为UTC时间，将其转为当地时间
now_local = now_utc.astimezone()
print(now_local)


# 将local time转为UTC时间
time_str = '2014-08-10 11:18:30'
now = datetime.strptime(time_str, time_format)
time_tuple = now.timetuple()
utc_now = mktime(time_tuple)
print(utc_now)
```

如果我们想要知道某个时区的某个时间对应于自己所在时区的时间，可以先将已知的时间转为UTC时间，再转为自己所在时区的时间。  
比如现在我要乘坐从旧金山到纽约的航班，已知飞机到达纽约的时间，求出该时间对应的旧金山的时间。

```python
import pytz
# 将到达纽约的时间转为UTC时间
arrival_nyc = '2014-05-01 23:33:24'
nyc_dt_naive = datetime.strptime(arrival_nyc, time_format)
eastern = pytz.timezone('US/Eastern')
nyc_dt = eastern.localize(nyc_dt_naive)
utc_dt = pytz.utc.normalize(nyc_dt.astimezone(pytz.utc))
print(utc_dt)


# 转为旧金山当地删减
pacific = pytz.timezone('US/Pacific')
sf_dt = pacific.normalize(utc_dt.astimezone(pacific))
print(sf_dt)
```





### 46.使用内置算法与数据结构

#### 双端队列

```python
from collections import deque
fifo = deque()
fifo.append(1)      # Producer
fifo.append(2)
fifo.append(3)
x = fifo.popleft()  # Consumer
print(x)

>>>
1
```

#### 有序字典

```python
from collections import OrderedDict
a = OrderedDict()
a['foo'] = 1
a['bar'] = 2

b = OrderedDict()
b['foo'] = 'red'
b['bar'] = 'blue'

for value1, value2 in zip(a.values(), b.values()):
    print(value1, value2)

>>>
(1, 'red')
(2, 'blue')
```

#### 带默认值的字典

```python
from collections import defaultdict
# 默认值为0的字典
stats = defaultdict(int)
stats['my_counter'] += 1
print(dict(stats))

>>>
{'my_counter': 1}
```

#### 堆队列

heapq模块提供了heappush,heappop,nsmallest等函数，能够在标准的list中创建对堆队列（默认最小堆）

```python
from heapq import *
a = []
heappush(a, 5)
heappush(a, 3)
heappush(a, 7)
heappush(a, 4)


print(heappop(a), heappop(a), heappop(a), heappop(a))

>>>
(3, 4, 5, 7)


a = []
heappush(a, 5)
heappush(a, 3)
heappush(a, 7)
heappush(a, 4)
assert a[0] == nsmallest(1, a)[0] == 3


print('Before:', a)
a.sort()
print('After: ', a)

>>>
('Before:', [3, 4, 7, 5])
('After: ', [3, 4, 5, 7])
```

```python
from bisect import bisect_left
i = bisect_left(x, 991234)
print(i)

>>>
991234


from timeit import timeit
print(timeit(
    'a.index(len(a)-1)',
    'a = list(range(100))',
    number=1000))
print(timeit(
    'bisect_left(a, len(a)-1)',
    'from bisect import bisect_left;'
    'a = list(range(10**6))',
    number=1000))

>>>
0.00200581550598
0.0013439655304
```

#### 与迭代器相关的工具

可以通过help(itertools)来查看内部的工具。


### 47.在重视精确度的场合，应该使用decimal

decimal模块中的Decimal可以实现数值的精确计算(直接计算虽然误差很小，比如真实值为3.45，而输出3.449999999，但是仍存在不便，而Decimal不会出现这种情况)和灵活的舍入方法。  
比如我想计算电话费，不足1分钱（0.01）的部分，要按照1分钱收。

```python
import decimal
from decimal import Decimal
rate = Decimal('1.45')
seconds = Decimal(str(3 * 60 + 41))
cost = rate * seconds / Decimal('60')
print cost

rounded = cost.quantize(Decimal('0.01'), rounding=decimal.ROUND_UP)
print rounded

>>>
5.340833333333333333333333333
5.35
```

### 48.学会安装由Python开发者社区所构建的模块