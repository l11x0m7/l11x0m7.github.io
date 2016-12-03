--- 
layout: post 
title: Leetcode的Hard难度题目汇总
date: 2016-12-03 
categories: blog 
tags: [算法, leetcode] 
description: 
--- 

# 149. Max Points on a Line

`Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.`

思路：  
算法的基本思路是遍历每个点，找到每个点对应的线段斜率，并统计斜率相同的线段数，从而得到结果。这里注意两个特殊情况：

* 两点在同一垂直线上
* 两点坐标重叠

复杂度：  
O(n^2logn)

代码：

```cpp
/**
 * Definition for a point.
 * struct Point {
 *     int x;
 *     int y;
 *     Point() : x(0), y(0) {}
 *     Point(int a, int b) : x(a), y(b) {}
 * };
 */
class Solution {
public:
    int maxPoints(vector<Point>& points) {
        int result = 0;
        for(int i=0;i<points.size();i++){
            unordered_map<double, int> m;
            int dup = 0, vertical = 0;
            double gradient = 0;
            int curmax = 0;
            for(int j=i+1;j<points.size();j++){
                if(points[i].x==points[j].x){
                    if(points[i].y==points[j].y)
                        dup++;
                    else
                        vertical++;
                    curmax = max(curmax, vertical);
                }
                else{
                    gradient = (points[i].y-points[j].y)*1.0/(points[i].x-points[j].x);
                    m[gradient]++;
                    curmax = max(curmax, m[gradient]);
                }
            }
            result = max(result, curmax+dup+1);
        }
        return result;
    }
};
```

