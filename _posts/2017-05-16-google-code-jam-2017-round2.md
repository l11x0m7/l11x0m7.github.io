--- 
layout: post 
title: Google Code Jam 2017——Round 2
date: 2017-05-16 
categories: blog 
tags: [算法] 
description: Google Code Jam 2017——Round 2
--- 

# Google Code Jam 2017——Round 2

题目地址：https://code.google.com/codejam/contest/5314486/dashboard

## A

### 思路

使用DP来做，因为考虑到给的数据中，一包巧克力的块数很少（[2, 4]），所以需要先统计每个旅游团过来后，开一包新的巧克力后剩余的巧克力数，即对于旅游团人数`N_i`，求`N_i mod P`，那么结果就只有`0, 1, ..., P-1`这`P`种情况。其中，模0的旅游团一定是包含在最优值内，所以可以单独先计算。  
建立`dp[i][j][k][m]`表示在模1剩余i个旅游团、模2剩余j个旅游团、模3剩余k个旅游团、前面剩余m块巧克力的时候，可能的只吃新的巧克力的旅游团的最大个数。同时建立状态方程(只考虑当前步骤取模1的旅游团，同理取模2和模3的)`dp[j-1][k][l][(m-1+P)%P] = max(dp[j][k][l][m] + int(m == 0), dp[j-1][k][l][(m-1+P)%P])`。

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
#include <map>  
using namespace std;

int T, N, P;
int dp[101][101][101][4];

int main()  
{  
    freopen("A-large.in.txt", "r", stdin);  
    //freopen("in.txt", "r",stdin);  
    freopen("out.txt", "w", stdout);  
    scanf("%d", &T);  
    for (int i = 1; i<= T; i++)  
    {  
        cin>>N>>P;
        int c[4] = {0};
        memset(c, 0, sizeof(c));
        memset(dp, 0, sizeof(dp));
        int people;
        for(int l=0;l<N;l++) {
            cin>>people;
            c[people % P]++;
        }
        dp[c[1]][c[2]][c[3]][0] = c[0];

        for(int j=c[1];j>=0;j--) {
            for(int k=c[2];k>=0;k--) {
                for(int l=c[3];l>=0;l--) {
                    for(int m=P-1;m>=0;m--) {
                        if(j>0)
                            dp[j-1][k][l][(m-1+P)%P] = max(dp[j][k][l][m] + int(m == 0), dp[j-1][k][l][(m-1+P)%P]);
                        if(k>0)
                            dp[j][k-1][l][(m-2+P)%P] = max(dp[j][k][l][m] + int(m == 0), dp[j][k-1][l][(m-2+P)%P]);
                        if(l>0)
                            dp[j][k][l-1][(m-3+P)%P] = max(dp[j][k][l][m] + int(m == 0), dp[j][k][l-1][(m-3+P)%P]);
                        // cout<<j<<" "<<k<<" "<<l<<" "<<m<<" "<<dp[j][k][l][m]<<endl;
                    }
                }
            }
        }
        printf("Case #%d: ", i);
        int res = max(max(max(dp[0][0][0][0], dp[0][0][0][1]), dp[0][0][0][2]), dp[0][0][0][3]);
        cout<<res<<endl;
    }  
    return 0;  
}  
```