--- 
layout: post 
title: C/C++常见问题汇总（不定期更新）
date: 2017-02-19 
categories: blog 
tags: [C++] 
description: C++编程的时候常见的问题
--- 

# C/C++常见问题汇总（不定期更新）

### 1. memset只能对int数组初始化为0或-1

memset只能够用来初始化char数组，而对int数组，则只会初始化0或-1。

原因是memset是一个字节一个字节的初始化的。  
char是一个字节，而int一般是4个字节。  
对于0，表示为 0x00000000，初始化后也为0x00000000  
对于-1，表示为 0xFFFFFFFF，初始化后也为0xFFFFFFFF  
对于1，表示为 0x00000001，但是初始化后为0x01010101

```cpp
char* a = {'a', 'b', 'c', 'd'};
memset(a, 'a', sizeof(a)); // 正确，因为char是一个字节

int* a = {1,2,3,4,5};
memset(a, 0, sizeof(a)); // 正确
memset(a, -1, sizeof(a)); // 正确
memset(a, 1, sizeof(a)); // 错误
```

### 2. const关键字 + 指针

const int a和int const a这两个写法是等同的，表示a是一个int常量。  
const int\* a和int const\* a表示a是一个指针，可以任意指向int常量或者int变量，它总是把它所指向的目标当作一个int常量，即不可以通过该指针来修改对应的int变量。  
int * const a表示a是一个指针常量，初始化的时候必须固定指向一个int变量，之后就不能再指向别的地方了。

例子：

```cpp
   int main()  
   {  
      int i = 12;   
      int const * p = &i; 
      p++;
      printf("%d\n",*p);   
      return 0;  
   }
// int const* 表示指向一个（自认为是）int常量，由于是p++，即在栈区位置移动sizeof(int)个字节，即地址发生改变，为0。

   int main()  
   {  
      int i = 12;   
      int* const p = &i; 
      p++;
      printf("%d\n",*p);   
      return 0;  
   }
// 这样会发生错误，因为int* const是一个指针常量，不能够改变该指针的值，但可以改变该指针所指向的变量i。
```

### 3. lower_bound和higher_bound的区别

对于某个已经**排序**的`vector<int> s`，其`lower_bound`和`upper_bound`分别取到以给定值为下界的最大值，只是包含和不包含的区别。

简单讲，就是`lower_bound`返回大于等于给定数的最小值，而`upper_bound`返回大于给定数的最小值。

### 4. 基本数据类型取值范围

unsigned int   0～4294967295   
int   			-2147483648～2147483647  
unsigned long 	0～4294967295  
long   -2147483648～2147483647  
long long的最大值：9223372036854775807  
long long的最小值：-9223372036854775808  
unsigned long long的最大值：1844674407370955161  
\_\_int64的最大值：9223372036854775807  
\_\_int64的最小值：-9223372036854775808  
unsigned __int64的最大值：18446744073709551615  

### 5. priority_queue的初始化

priority_queue接受三个初始化参数，分别为`存储单元基本类型、存储容器、比较函数`，比如如果要存储的基本单元为`int`，容器为`vector`，使用默认比较函数（默认为operator<，即把大的放堆顶），则可以初始化为`priority_queue<int, vector<int>, less<int>> pq`。

对于自定义的比较函数初始化方法，可以是对自定义类重载小于操作符即（`<`），也可以是重载函数操作符对象（即`()`），也可以使用lambda函数。现在主要介绍后面两种常用的方法。

```cpp
// 重载函数操作符对象

struct mycmp{
    bool operator()(vector<int> a, vector<int> b){
        return a[0] < b[0];
    }
};

priority_queue<vector<int>, vector<vector<int>>, mycmp> pq;

// 匿名函数

auto cmp = [](vector<int> a, vector<int> b){return a[0] < b[0];};

priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> pq(cmp);

priority_queue<vector<int>, vector<vector<int>>, function<bool(vector<int>&, vector<int>&)>> pq(cmp);
```


### 6. vector直接用数组初始化

从[cplusplus](http://www.cplusplus.com/reference/vector/vector/vector/)上得到的初始化例子如下：

```cpp
// constructing vectors
#include <iostream>
#include <vector>

int main ()
{
  // constructors used in the same order as described above:
  std::vector<int> first;                                // empty vector of ints
  std::vector<int> second (4,100);                       // four ints with value 100
  std::vector<int> third (second.begin(),second.end());  // iterating through second
  std::vector<int> fourth (third);                       // a copy of third

  // the iterator constructor can also be used to construct from arrays:
  int myints[] = {16,2,77,29};
  std::vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );

  std::cout << "The contents of fifth are:";
  for (std::vector<int>::iterator it = fifth.begin(); it != fifth.end(); ++it)
    std::cout << ' ' << *it;
  std::cout << '\n';

  return 0;
}
```

可以直接用数组初始化：

```cpp
vector<int> vec({1,2,3,4});
```