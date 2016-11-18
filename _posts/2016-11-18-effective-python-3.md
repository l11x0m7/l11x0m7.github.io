--- 
layout: post 
title: 《Effective Python》读书笔记(三) 类和继承
date: 2016-11-18 
categories: blog 
tags: [Python] 
description: effective python读书笔记3
--- 

# 《Effective Python》读书笔记（三） 类和继承

### 23.尽量使用挂钩函数来作为简单接口的接收函数

挂钩函数就相当于如下形式所示。

```python
def log_missing():
    print 'Add key'
    return 0
current = {'green':12, 'blue':3}
increments = [('red', 5), ('blue', 17), ('orange', 9)]
# log_missing为挂钩函数
result = collections.defaultdict(log_missing, current)
print dict(result)
for key, amount in increments:
    result[key] += amount
print dict(result)

{'blue': 3, 'green': 12}
Add key
Add key
{'blue': 20, 'orange': 9, 'green': 12, 'red': 5}
```

如果需要记录状态数，可以创建一个辅助类。

```python
current = {'green':12, 'blue':3}
increments = [('red', 5), ('blue', 17), ('orange', 9)]
# 如果要记录状态值,可以用辅助类
class CountMissing(object):
    def __init__(self):
        self.added = 0
    def missing(self):
        self.added += 1
        return 0
cm = CountMissing()
result = collections.defaultdict(cm.missing, current)
print dict(result)
for key, amount in increments:
    result[key] += amount
print dict(result)
    
>>>
{'blue': 3, 'green': 12}
{'blue': 20, 'orange': 9, 'green': 12, 'red': 5}
```

当然也可以使用__call__函数，来让类的实例也能够像函数一样被调用。当使用callable()时，会返回True。

```python
current = {'green':12, 'blue':3}
increments = [('red', 5), ('blue', 17), ('orange', 9)]
# 如果要记录状态值,也可以这么写,用__call__
class CountMissing(object):
    def __init__(self):
        self.added = 0
    def __call__(self, *args, **kwargs):
        self.added += 1
        return 0

cm = CountMissing()
result = collections.defaultdict(cm, current)
print dict(result)
for key, amount in increments:
    result[key] += amount
print dict(result)
print callable(cm)

>>>
{'blue': 3, 'green': 12}
{'blue': 20, 'orange': 9, 'green': 12, 'red': 5}
True
```

### 24.用@classmethod形式的多态去通用的构建对象

@classmethod放在类中某个方法前，将其变为类方法，从而可以直接被类调用。通过@classmethod可以实现多态，即在调用函数的时候，可以直接将类作为参数传递，而在函数内部可以直接调用类方法。

下面通过MapReduce的例子来看。首先看一下不用@classmethod的情况：

```python
# Example 1
class InputData(object):
    def read(self):
        raise NotImplementedError


# Example 2
class PathInputData(InputData):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def read(self):
        return open(self.path).read()


# Example 3
class Worker(object):
    def __init__(self, input_data):
        self.input_data = input_data
        self.result = None

    def map(self):
        raise NotImplementedError

    def reduce(self, other):
        raise NotImplementedError


# Example 4
class LineCountWorker(Worker):
    def map(self):
        data = self.input_data.read()
        self.result = data.count('\n')

    def reduce(self, other):
        self.result += other.result


# Example 5
import os

def generate_inputs(data_dir):
    for name in os.listdir(data_dir):
        yield PathInputData(os.path.join(data_dir, name))


# Example 6
def create_workers(input_list):
    workers = []
    for input_data in input_list:
        workers.append(LineCountWorker(input_data))
    return workers


# Example 7
from threading import Thread

def execute(workers):
    threads = [Thread(target=w.map) for w in workers]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    first, rest = workers[0], workers[1:]
    for worker in rest:
        first.reduce(worker)
    return first.result


# Example 8
def mapreduce(data_dir):
    inputs = generate_inputs(data_dir)
    workers = create_workers(inputs)
    return execute(workers)


# Example 9
from tempfile import TemporaryDirectory
import random

def write_test_files(tmpdir):
    for i in range(100):
        with open(os.path.join(tmpdir, str(i)), 'w') as f:
            f.write('\n' * random.randint(0, 100))

with TemporaryDirectory() as tmpdir:
    write_test_files(tmpdir)
    result = mapreduce(tmpdir)

print('There are', result, 'lines')

>>>
There are 4098 lines
```

上面的Worker这一大类和InputData这一大类需要通过mapreduce这个方法连接起来，并且如果我需要创建新的Worker或InputData，那么扩展起来就不太方便了。

现在看看使用@classmethod来实现多态。

```python
# Example 10
class GenericInputData(object):
    def read(self):
        raise NotImplementedError

    @classmethod
    def generate_inputs(cls, config):
        raise NotImplementedError


# Example 11
class PathInputData(GenericInputData):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def read(self):
        return open(self.path).read()

    @classmethod
    def generate_inputs(cls, config):
        data_dir = config['data_dir']
        for name in os.listdir(data_dir):
            yield cls(os.path.join(data_dir, name))


# Example 12
class GenericWorker(object):
    def __init__(self, input_data):
        self.input_data = input_data
        self.result = None

    def map(self):
        raise NotImplementedError

    def reduce(self, other):
        raise NotImplementedError

    @classmethod
    def create_workers(cls, input_class, config):
        workers = []
        for input_data in input_class.generate_inputs(config):
            workers.append(cls(input_data))
        return workers


# Example 13
class LineCountWorker(GenericWorker):
    def map(self):
        data = self.input_data.read()
        self.result = data.count('\n')

    def reduce(self, other):
        self.result += other.result


# Example 14
def mapreduce(worker_class, input_class, config):
    workers = worker_class.create_workers(input_class, config)
    return execute(workers)


# Example 15
with TemporaryDirectory() as tmpdir:
    write_test_files(tmpdir)
    config = {'data_dir': tmpdir}
    result = mapreduce(LineCountWorker, PathInputData, config)
print('There are', result, 'lines')

>>>
There are 4098 lines
```

> 在python中，每个类只能有一个构造器，即一个__init__方法
> 可以用@classmethod来仿造构造器，从而构造类的对象


### 25.用super初始化父类

直接调用类的__init__函数来初始化父类，那么父类的初始化顺序是按照子类的__init__里对各个超类的__init__调用顺序来进行。  
下面看一下不用super来初始化的时候，在钻石型继承中出现的问题。

```python
class MyBaseClass(object):
    def __init__(self, value):
        self.value = value

class TimesFive(MyBaseClass):
    def __init__(self, value):
        MyBaseClass.__init__(self, value)
        self.value *= 5

class PlusTwo(MyBaseClass):
    def __init__(self, value):
        MyBaseClass.__init__(self, value)
        self.value += 2
# 多重继承，即继承多个父类
class ThisWay(TimesFive, PlusTwo):
    def __init__(self, value):
        TimesFive.__init__(self, value)
        PlusTwo.__init__(self, value)

foo = ThisWay(5)
print 'Should be (5*5)+2 but is', foo.value

>>>
Should be (5*5)+2 but is 7
```

由于TimesFive和PlusTwo在调用__init__的时候都各自调用了一次MyBaseClass的__init__，因此为7。显然，这种方式是错误的。

现在考虑使用super，其定义了“方法解析顺序”（method resolution order，MRO）。MRO是以标准的流程来安排超类的初始化顺序（深度优先，从左往右）。

```python
# python2风格
class MyBaseClass(object):
    def __init__(self, value):
        self.value = value

class TimesFive(MyBaseClass):
    def __init__(self, value):
        super(TimesFive, self).__init__(value)
        self.value *= 5

class PlusTwo(MyBaseClass):
    def __init__(self, value):
        super(PlusTwo, self).__init__(value)
        self.value += 2

class ThisWay(TimesFive, PlusTwo):
    def __init__(self, value):
        super(ThisWay, self).__init__(value)

foo = ThisWay(5)
print 'Should be (5*5)+2 and it is', foo.value

# 调用顺序
from pprint import pprint
pprint(ThisWay.mro())
    
# python3风格
class MyBaseClass(object):
    def __init__(self, value):
        self.value = value

class TimesFive(MyBaseClass):
    def __init__(self, value):
        # 或者super().__init__(value)
        super(__class__, self).__init__(value)
        self.value *= 5

class PlusTwo(MyBaseClass):
    def __init__(self, value):
        super(__class__, self).__init__(value)
        self.value += 2

class ThisWay(TimesFive, PlusTwo):
    def __init__(self, value):
        super().__init__(value)

foo = ThisWay(5)
print 'Should be (5*5)+2 and it is', foo.value

# 调用顺序
from pprint import pprint
pprint(ThisWay.mro())


>>>
Should be (5*(5+2)) ant it is 35
[<class '__main__.ThisWay'>,
 <class '__main__.TimesFive'>,
 <class '__main__.PlusTwo'>,
 <class '__main__.MyBaseClass'>,
 <type 'object'>]
```

输出结果为35，是正确的。使用super，其初始化的逻辑顺序为：要初始化ThisWay，先初始化TimesFive（深度优先）；要初始化TimesFive，先初始化PlusTwo（从左到右）；要初始化PlusTwo，先初始化MyBaseClass（深度优先）。所以初始化顺序为：  
`MyBaseClass->PlusTwo->TimesFive->ThisWay`。


### 26.只在使用mix-in组件制作工具类时进行多重继承

mix-in是指只实现了单个功能（方法）的类，或者继承这些类的类。  
下面以用于类的序列化的mix-in组件为例。

> isinstance函数可以动态检测对象类型
> __dict__可以打印类实例的所有成员值和类实例的默认私有成员值（如__module__等），并以键值对的形成出现
> hasattr函数可以判定某个类实例里有没有某个成员或方法

```python
class ToDictMixin(object):
    def to_dict(self):
        return self._traverse_dict(self.__dict__)
    def _traverse_dict(self, instance_dict):
        output = {}
        for key, value in instance_dict.iteritems():
            cur = self._traverse(key, value)
            # 不现实空值
            if cur is not None:
                output[key] = cur
        return output
    def _traverse(self, key, value):
        # 当然也可以写成isinstance(value, BinaryTree),但是为了通用性,一般写父类
        if isinstance(value, ToDictMixin):
            return value.to_dict()
        # 下面三个该例子中没有用到,可以注释掉
        # elif isinstance(value, dict):
        #     return self._traverse_dict(value)
        # elif isinstance(value, list):
        #     return [self._traverse(key, i) for i in value]
        # elif hasattr(value, '__dict__'):
        #     return self._traverse_dict(value.__dict__)
        else:
            return value

class BinaryTree(ToDictMixin):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

tree = BinaryTree(10, left=BinaryTree(7,right=BinaryTree(10)), 
right=BinaryTree(3, left=BinaryTree(11)))

print tree.to_dict()
    
>>>
{'right': {'value': 3, 'left': {'value': 11}}, 
'value': 10, 
'left': {'right': {'value': 10}, 'value': 7}}
```

下面实现一下多个mix-in的搭配。比如使用JsonMixin来**load成json字符串**，然后将加载后的值**初始化**（相当于反序列化，deserialize）DatacenterRack对象，再将该对象通过ToDictMixin来进行**序列化**，之后再**dump成json字符串**。


```python
import json

class JsonMixin(object):
    @classmethod
    def from_json(cls, data):
        kwargs = json.loads(data)
        return cls(**kwargs)

    def to_json(self):
        return json.dumps(self.to_dict())


class DatacenterRack(ToDictMixin, JsonMixin):
    def __init__(self, switch=None, machines=None):
        self.switch = Switch(**switch)
        self.machines = [
            Machine(**kwargs) for kwargs in machines]

class Switch(ToDictMixin, JsonMixin):
    def __init__(self, ports=None, speed=None):
        self.ports = ports
        self.speed = speed

class Machine(ToDictMixin, JsonMixin):
    def __init__(self, cores=None, ram=None, disk=None):
        self.cores = cores
        self.ram = ram
        self.disk = disk


serialized = """{
    "switch": {"ports": 5, "speed": 1e9},
    "machines": [
        {"cores": 8, "ram": 32e9, "disk": 5e12},
        {"cores": 4, "ram": 16e9, "disk": 1e12},
        {"cores": 2, "ram": 4e9, "disk": 500e9}
    ]
}"""

deserialized = DatacenterRack.from_json(serialized)
roundtrip = deserialized.to_json()
assert json.loads(serialized) == json.loads(roundtrip)
```

### 27.多用public属性，少用private属性

各个属性值的含义：

> self.field表示public成员
> self._field表示protect成员
> self.__field表示私有成员，它可以被类内部方法访问，在类外，可以通过instance._Class__field被访问，Class就是该类对应的名称。因此python无法保证private成员的私密性。
> self.__len___()表示类中的特殊成员或方法

一般情况下，不要在类内定义private成员，应多用protect代替。在下面一种情况下，可以使用private，来防止子类的属性覆盖同名的超类属性。

```python
class ApiClass(object):
    def __init__(self):
        self._value = 5

    def get(self):
        return self._value

class Child(ApiClass):
    def __init__(self):
        super().__init__()
        self._value = 'hello'  # Conflicts

a = Child()
print(a.get(), 'and', a._value, 'should be different')

>>>
hello and hello should be different

class ApiClass(object):
    def __init__(self):
        self.__value = 5

    def get(self):
        return self.__value

class Child(ApiClass):
    def __init__(self):
        super().__init__()
        self._value = 'hello'  # OK!

a = Child()
print(a.get(), 'and', a._value, 'are different')

>>>
5 and hello are different
```

### 28.继承collections.abc（在python3里有）以实现自定义的容器类型

collections.abc中定义了很多容器的抽象基类，如果要自定义容器，最好就是继承需要的抽象基类，然后实现抽象基类当中的某些特殊方法（如__getitems__和__len__都是特殊方法），那么自定义类就具备了抽象基类提供的其他方法，如count和index方法。  
下面给出一个使用collections.abc中的Sequence抽象基类实现自定义容器的实例。

> 索引访问（`比如foo[0]`）其实就是调用`__getitem__()`方法（比如`foo.__getitem__(0)`）
> 使用`len(a)`相当于调用`a.__len__()`

```python
# 实现__getitem__()方法
class IndexableNode(BinaryNode):
	# 前序遍历
    def _search(self, count, index):
        found = None
        if self.left:
            found, count = self.left._search(count, index)
        if not found and count == index:
            found = self
        else:
            count += 1
        if not found and self.right:
            found, count = self.right._search(count, index)
        return found, count
        # Returns (found, count)

    def __getitem__(self, index):
        found, _ = self._search(0, index)
        if not found:
            raise IndexError('Index out of range')
        return found.value

# 实现__len__()方法
class SequenceNode(IndexableNode):
    def __len__(self):
        _, count = self._search(0, None)
        return count

# 载入模块
from collections.abc import Sequence

class BetterNode(SequenceNode, Sequence):
    pass

tree = BetterNode(
    10,
    left=BetterNode(
        5,
        left=BetterNode(2),
        right=BetterNode(
            6, right=BetterNode(7))),
    right=BetterNode(
        15, left=BetterNode(11))
)

print('Index of 7 is', tree.index(7))
print('Count of 10 is', tree.count(10))

>>>
Index of 7 is 3
Count of 10 is 1
```

当然，如果自定义的容器比较简单，可以直接继承像list、dict、set这样的类，然后加入自己的方法。实例如下：

```python
class FrequencyList(list):
    def __init__(self, data):
        super(FrequencyList, self).__init__(data)

    def frequency(self):
        count = collections.defaultdict(lambda:0, {})
        for item in self:
            count[item] += 1
        return dict(count)

fl = FrequencyList(['a', 'b', 'c', 'c', 'a', 'd', 'f', 'b'])
print repr(fl.frequency())

>>>
{'a': 2, 'c': 2, 'b': 2, 'd': 1, 'f': 1}
```