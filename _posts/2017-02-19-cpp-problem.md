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

