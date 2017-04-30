--- 
layout: post 
title: Google Code Jam 2017——Round 1C
date: 2017-04-30 
categories: blog 
tags: [算法] 
description: google code jam, round 1C
--- 

# Google Code Jam 2017——Round 1C

题目地址：https://code.google.com/codejam/contest/3274486/dashboard

## A


### 思路

贪婪法。先按照半径升序排序，找到每个半径放在最下面的情况下的总面积。每次挪到下一个的时候，删除前一个高度最小的，并加入当前的pancake作为底。

### 代码

```cpp
#include <iostream>
#include <cstdio>  
#include <iostream>  
#include <string>  
#include <iterator>  
#include <algorithm>  
#include <vector>  
#include <cstring>  
#include <array>  
#include <queue>  
#include <set>  
#include <cmath>
#include <map>  
#include <iomanip>
using namespace std;

int N, K;
#define PI 3.14159265358979323846

struct cmp {
    bool operator() (vector<double> a, vector<double> b) {
        return a[0] < b[0];
    }
} comp;

double solve(vector<vector<double> >& p) {
    sort(p.begin(), p.end(), comp);
    priority_queue<double, vector<double> > pq;
    double res = 0.;
    for(int i=0;i<K;i++) {
        pq.push(-p[i][0] * p[i][1]);
        res += p[i][0] * p[i][1];
    }
    res += pow(p[K-1][0], 2.) / 2.;
    double total_res = res;
    for(int i=K;i<N;i++) {
        res -= pow(p[i-1][0], 2) / 2.;
        res += pow(p[i][0], 2) / 2;
        res -= -pq.top();
        pq.pop();
        double mul = p[i][0] * p[i][1];
        pq.push(-mul);
        res += mul;
        total_res = max(total_res, res);
    }
    return total_res * 2. * PI;
}




int main()  
{  
    freopen("A-large-practice.in", "r", stdin);  
    //freopen("in.txt", "r",stdin);  
    freopen("out.txt", "w", stdout);  
    int t;  
    scanf("%d", &t);  
    string val;
    for (int i = 1; i<= t; i++)  
    {  
        cin>>N>>K;
        vector<vector<double> > p(N, vector<double>(2, 0.));
        for(int j=0;j<N;j++)
            cin>>p[j][0]>>p[j][1];
        double res = solve(p);
        printf("Case #%d: ", i);
        cout<<setiosflags(ios::fixed)<<setprecision(9)<<res<<endl;
    }  
    return 0;  
}  
```


## B

### 思路

`dp[i][j][k]`表示C用了`i`（`0<=i<=720`）分钟，J用了`j`（`0<=j<=720`）分钟，且当前为`k`（`k==0 || k==1`）在照顾。注意分开讨论，如果一开始是C照顾，则结尾还是C照顾的话，两段会合并成一段。同理，如果一开始是J照顾，则结尾还是J的话，两段会合并成一段。


### 代码

```cpp
#include <iostream>
#include <cstdio>  
#include <iostream>  
#include <string>  
#include <iterator>  
#include <algorithm>  
#include <vector>  
#include <cstring>  
#include <array>  
#include <queue>  
#include <set>  
#include <cmath>
#include <map>  
#include <iomanip>
using namespace std;

const int TIME = 720;
const int PERSON = 2;
int dp[TIME + 1][TIME + 1][PERSON];
int C, J;

int solve(vector<int>& bit) {
    for(int i=0;i<=TIME;i++) {
        for(int j=0;j<=TIME;j++)
            for(int k=0;k<PERSON;k++)
                dp[i][j][k] = INT_MAX;
    }
    dp[0][0][0] = 0;
    // dp[0][0][1] = 0;
    for(int i=0;i<=TIME;i++) {
        for(int j=0;j<=TIME;j++) {
            if(i == 0 && j == 0)
                continue;
            if(bit[i + j] == 0) {
                if(i != 0) {
                    dp[i][j][0] = dp[i-1][j][0];
                    if(dp[i-1][j][1] != INT_MAX)
                        dp[i][j][0] = min(dp[i][j][0], dp[i-1][j][1] + 1);
                }
                if(j != 0) {
                    dp[i][j][1] = dp[i][j-1][1];
                    if(dp[i][j-1][0] != INT_MAX)
                        dp[i][j][1] = min(dp[i][j][1], dp[i][j-1][0] + 1);
                }
            }
            else if(bit[i + j] == 1) {
                if(i == 0)
                    continue;
                dp[i][j][0] = dp[i-1][j][0];
                if(dp[i-1][j][1] != INT_MAX)
                    dp[i][j][0] = min(dp[i][j][0], dp[i-1][j][1] + 1);
            }
            else {
                if(j == 0)
                    continue;
                dp[i][j][1] = dp[i][j-1][1];
                if(dp[i][j-1][0] != INT_MAX)
                    dp[i][j][1] = min(dp[i][j][1], dp[i][j-1][0] + 1);
            }
        }
    }
    int ans1 = dp[TIME][TIME][0];
    if(dp[TIME][TIME][1] != INT_MAX)
        ans1 = min(ans1, dp[TIME][TIME][1] + 1);
    for(int i=0;i<=TIME;i++) {
        for(int j=0;j<=TIME;j++)
            for(int k=0;k<PERSON;k++)
                dp[i][j][k] = INT_MAX;
    }
    dp[0][0][1] = 0;
    for(int i=0;i<=TIME;i++) {
        for(int j=0;j<=TIME;j++) {
            if(i == 0 && j == 0)
                continue;
            if(bit[i + j] == 0) {
                if(i != 0) {
                    dp[i][j][0] = dp[i-1][j][0];
                    if(dp[i-1][j][1] != INT_MAX)
                        dp[i][j][0] = min(dp[i][j][0], dp[i-1][j][1] + 1);
                }
                if(j != 0) {
                    dp[i][j][1] = dp[i][j-1][1];
                    if(dp[i][j-1][0] != INT_MAX)
                        dp[i][j][1] = min(dp[i][j][1], dp[i][j-1][0] + 1);
                }
            }
            else if(bit[i + j] == 1) {
                if(i == 0)
                    continue;
                dp[i][j][0] = dp[i-1][j][0];
                if(dp[i-1][j][1] != INT_MAX)
                    dp[i][j][0] = min(dp[i][j][0], dp[i-1][j][1] + 1);
            }
            else {
                if(j == 0)
                    continue;
                dp[i][j][1] = dp[i][j-1][1];
                if(dp[i][j-1][0] != INT_MAX)
                    dp[i][j][1] = min(dp[i][j][1], dp[i][j-1][0] + 1);
            }
        }
    }
    int ans2 = dp[TIME][TIME][1];
    if(dp[TIME][TIME][0] != INT_MAX)
        ans2 = min(ans2, dp[TIME][TIME][0] + 1);
    return min(ans1, ans2);
}

int main()  
{  
    freopen("B-large-practice.in", "r", stdin);  
    //freopen("in.txt", "r",stdin);  
    freopen("out.txt", "w", stdout);  
    int t;  
    scanf("%d", &t);  
    for (int i = 1; i<= t; i++)  
    {  
        cin>>C>>J;
        vector<int> bit(2 * TIME + 1, 0);
        int b, e;
        for(int j=0;j<C;j++) {
            cin>>b>>e;
            for(int k=b;k<e;k++)
                bit[k] = 1;
        }
        for(int j=0;j<J;j++) {
            cin>>b>>e;
            for(int k=b;k<e;k++)
                bit[k] = 2;
        }
        int res = solve(bit);
        printf("Case #%d: ", i);
        cout<<res<<endl;
    }  
    return 0;  
}  
```