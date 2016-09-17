--- 
layout: post 
title: Reservoir sampling
date: 2016-09-17 
categories: blog 
tags: [leetcode, 算法] 
description: 水库采样
--- 

# Reservoir sampling

> Reservoir sampling is a family of randomized algorithms for randomly choosing a sample of k items from a list S containing n items, where n is either a very large or unknown number. Typically n is large enough that the list doesn't fit into main memory.

>——Wiki

存储采样是指从一个包含n个对象的列表S中随机抽取出k个对象作为样本，n要么很大要么未知。典型的n通常大到无法将整个列表存入主内存。

这里，我们主要考虑如何通过这个思路产生我们想要的随机样本。通常，在考虑选取数据的时候，会按照等概选取。我们接下来就针对等概选取来说明。

假设有n个对象，我们要从中等概选取k个对象，步骤如下：

* 先将第一个对象放入内存，即选中该对象
* 对每一个后面的对象<span>$$$i$$$</span>
	* 有1/i的概率会用新值覆盖旧的值
	* 有1-1/i的概率会丢弃新的值
* 根据上面的情况，如果总共有k个对象，那么
	* 对象1会以概率<img src="http://chart.googleapis.com/chart?cht=tx&chl=\frac12*\frac23*\ldots\frac{n-1}n=\frac{1}{n}" style="border:none;">被选中
	* 对象2会以概率<img src="http://chart.googleapis.com/chart?cht=tx&chl=1*\frac12*\frac23*\ldots\frac{n-1}n=\frac{1}{n}" style="border:none;">被选中
	* 对象3会以概率<img src="http://chart.googleapis.com/chart?cht=tx&chl=\frac13*\ldots\frac{n-1}n=\frac{1}{n}" style="border:none;">被选中

综上，可以看到，对于序列中的n个对象，均会以等概方式被选到。

现在通过两个leetcode的例子来看下。

### 1.Random Pick Index


Given an array of integers with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

**Note:**
The array size can be very large. Solution that uses too much extra space will not pass the judge.

**Example:**

```
int[] nums = new int[] {1,2,3,3,3};
Solution solution = new Solution(nums);

// pick(3) should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(3);

// pick(1) should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(1);
```
我的C++代码如下：

```cpp
class Solution {
public:
    vector<int> nums;
    Solution(vector<int> nums) {
        this->nums = nums;
        srand(NULL);
    }
    int pick(int target) {
        int cnt = 0;
        int ind = -1;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==target){
                cnt++;
                if(ind==-1)
                    ind = i;
                else if(rand()%cnt==0)
                    ind = i;
            }
        }
        return ind;
    }
    };
/**
 \* Your Solution object will be instantiated and called as such:
 \* Solution obj = new Solution(nums);
 \* int param_1 = obj.pick(target);
 \*/
```

上面这题其实就是考虑如何从n中随机取出某个值为target的下标，可以按照Reservoir Sampling的思路计算。

> 注：此时我们假设C++中的rand()%n函数是完全随机的，实则不是。因为rand()是一个有上限的数，它会令小一些的数更高概率出现。

### 2.Linked List Random Node

Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen.

**Follow up:**
What if the linked list is extremely large and its length is unknown to you? Could you solve this efficiently without using extra space?

**Example:**

```
// Init a singly linked list [1,2,3].
ListNode head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
Solution solution = new Solution(head);

// getRandom() should return either 1, 2, or 3 randomly. Each element should have equal probability of returning.
solution.getRandom();
```

我的C++代码：

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    /** @param head The linked list's head. Note that the head is guanranteed to be not null, so it contains at least one node. */
    Solution(ListNode* head) {
        h = head;
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        int rd = 1;
        ListNode* tmp = h;
        ListNode* res = NULL;
        while(tmp){
            if(rand()%rd==0)
                res = tmp;
            tmp = tmp->next;
            rd++;
        }
        return res->val;
    }
private:
    ListNode* h;
};

/**
 \* Your Solution object will be instantiated and called as such:
 \* Solution obj = new Solution(head);
 \* int param_1 = obj.getRandom();
 \*/
```

思路类似，不做过多赘述。