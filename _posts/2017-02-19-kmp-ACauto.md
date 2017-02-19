--- 
layout: post 
title: KMP算法与AC自动机
date: 2017-02-19 
categories: blog 
tags: [算法] 
description: 两种模式匹配的算法
--- 

# KMP算法与AC自动机

KMP算法——用于单模匹配。  
AC自动机——用于多模匹配，需要了解KMP原理和Trie树。

### KMP算法

KMP算法用于单模匹配，比如在一个目标串当中匹配一个模式串。暴力解法就是扫描目标串与模式串，如果发现不匹配，则目标串起始点回溯到原起始点，再后移一位，而模式串回溯到开头0。  
KMP算法能够高效的找出目标串中的匹配串，相比于暴力搜索的O(m*n)的时间复杂度，KMP算法的搜索复杂度为O(m+n)。  

#### KMP算法的基本原理

##### 先求出next数组（数组下标表示模式子串长度，数组值表示该模式子串对应的最大公共子串长度）

举个例子，如果我们的目标串为S，匹配串为P，那么在匹配过程中，于$S_i$处失配：  

$$S_0,S_1,S_2,S_3,...,S_{i-j},S_{i-j+1},...,S_i,S_{i+1},...,S_m$$

$$P_0,P_1,...,P_{j-1},P{j},...$$

现在假设向右移动一位模式串，此时恰好又和目标串匹配上，即为：

![p1]()

这说明$P_0$到$P_{j-1}$是匹配的，那么我们可以在这个P的子串中找到与该子串后缀一致的前缀，比如子串"abcab"中，前缀与后缀匹配的最大公共子串为"ab"，则我们可以将串P进行移动，使得前缀与原来的后缀对齐。

结论：当发生失配的情况下，j的新值next[j]取决于模式串中P[0 ~ j-1]中前缀和后缀相等部分的长度， 并且next[j]恰好等于这个最大长度。  

##### 通过search查找目标串中的模式串

对比S和P，如果`S[i] == S[j]`，则向右移动；如果`S[i] != P[j]`，则使得`j = next[j]`，直到`S[i] == S[j]`或者`j == 0`（即模式子串的最大公共长度为0）。当`j == P.length()`的时候，表示在目标串中找到了一个匹配串。

#### 代码实现

```cpp
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <stack>
#include <cstdio>
#include <string>
#include <stdio.h>

using namespace std;

int* getNext(string p) {
	// 下标为子串长度，存储的值为最大公共长度
	int* next = new int[p.length()+1];
	// i和j都表示下标
	int j = 0;
	// 最大公共长度是指子串前缀和后缀的最大公共长度。如子串'abca'的最大公共长度为1
	next[0] = next[1] = 0; // 长度为0和1的子串的最大公共长度为0(比如'abcab'，长度为0时是""，长度为1时是"a")
	// 遍历每个子串长度
	for(int i=1;i<p.length();i++){
		// 寻找最大的公共子串
		while(j > 0 && p[i] != p[j])j = next[j];
		// 如果对于p[i]和最大公共子串后第一个字符p[j]相同，则
		if(p[i] == p[j])j++;
		next[i+1] = j;
	}
	return next;
}

void search(string s, string p) {
	int* next = getNext(p);
	int j = 0;
	for(int i=0;i<s.length();i++) {
		while(j > 0 && s[i] != p[j]) j = next[j];
		if(s[i] == p[j])	
			j++;
		if(j == p.length()) {
			cout<<"The match position is "<<i-j+1<<" to "<<i<<endl;
			j = next[j];
		}
	}
}

   int main()  
   {  
      string a = "abcdcabc";
      string b = "bc";
      search(a, b);
   }
```

### AC自动机

待补充……

### 参考

1. [KMP算法之总结篇](http://www.cnblogs.com/mfryf/archive/2012/08/15/2639565.html)
2. [KMP算法详解](http://blog.csdn.net/yutianzuijin/article/details/11954939/)