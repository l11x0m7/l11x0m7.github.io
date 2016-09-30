--- 
layout: post 
title: 双端队列的妙用——滑动窗口的最大值
date: 2016-09-30 
categories: blog 
tags: [lintcode] 
description: 双端队列
--- 

# 双端队列的妙用——滑动窗口的最大值

双端队列deque比较灵活，它融合了队列queue和栈stack的特性，数据可以从前面进出，也可以从后面进出。C++中的deque具体用法可以参考[cpluscplus](http://www.cplusplus.com/reference/deque/deque/?kw=deque)

下面我们通过一道来自《剑指Offer》和lintcode的题来看如何使用双端队列。

## 滑动窗口的最大值

```
给出一个可能包含重复的整数数组，和一个大小为 k 的滑动窗口, 从左到右在数组中滑动这个窗口，找到数组中每个窗口内的最大值。

样例
给出数组 [1,2,7,7,8], 滑动窗口大小为 k = 3. 返回 [7,7,8].

解释：

最开始，窗口的状态如下：

[|1, 2 ,7| ,7 , 8], 最大值为 7;

然后窗口向右移动一位：

[1, |2, 7, 7|, 8], 最大值为 7;

最后窗口再向右移动一位：

[1, 2, |7, 7, 8|], 最大值为 8.
```

思路：  
考虑一个双端队列deque，用于保存nums的index，并保证deque中的所有数都在窗口里，且从队头到队尾按从大到小排列。如果后面有新的数进来，则删除deque中比这个数小的数（因为这些在deque中的数都是排在当前数的前面，且在窗口内，如果当前数一直在窗口内，则最大值至少为这个数，而不可能是前面那些数，所以直接删除掉）。

代码：
 
```cpp
class Solution {
public:
    /**
     * @param nums: A list of integers.
     * @return: The maximum number inside the window at each moving.
     */
    vector<int> maxSlidingWindow(vector<int> &nums, int k) {
        // write your code here
        int len = nums.size();
        deque<int> dq;
        vector<int> res;
        if(len<1||k>len)
            return res;
        for(int i=0;i<len;i++){
            if(i<k){
                while(!dq.empty()&&nums[dq.back()]<nums[i])
                    dq.pop_back();
                dq.push_back(i);
            }
            else{
                res.push_back(nums[dq.front()]);
                while(!dq.empty()&&nums[dq.back()]<nums[i])
                    dq.pop_back();
                dq.push_back(i);
                if(!dq.empty()&&i-k+1>dq.front())
                    dq.pop_front();
            }
        }
        res.push_back(nums[dq.front()]);
        return res;
    }
};
```