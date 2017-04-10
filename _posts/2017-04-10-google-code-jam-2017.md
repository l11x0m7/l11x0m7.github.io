--- 
layout: post 
title: 
date: 2017-04-10 
categories: blog 
tags: [, ] 
description: google code jam
--- 

# Google Code Jam 2017——资格赛

题目地址：https://code.google.com/codejam/contest/3264486/dashboard


只写了前面三题，比较水，直接给代码。


## A

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


string solve(string& s, int size, int count) {
    int flip = 0;
    for(int i=0;i<=s.size() - size;i++) {
        if(s[i] == '-') {
            int count_neg = 0, count_pos = 0;
            for(int j=i;j<i+size;j++) {
                if(s[j] == '-') {
                    s[j] = '+';
                    count_neg++;
                }
                else {
                    s[j] = '-';
                    count_pos++;
                }
            }
            flip++;
            count += (count_neg - count_pos);
        }
    }
    if(count == s.size())
        return to_string(flip);
    else
        return "IMPOSSIBLE";

}

// char val[1010] = {0};

int main()  
{  
    freopen("A-large.in.txt", "r", stdin);  
    //freopen("in.txt", "r",stdin);  
    freopen("out.txt", "w", stdout);  
    int t;  
    scanf("%d", &t);  
    int size;
    string val;
    for (int i = 1; i<= t; i++)  
    {  
        cin>>val;
        cin>>size;
        // count for +
        int count = 0;
        for(auto c : val)
            if(c == '+')
                count++;
        printf("Case #%d: ", i);
        string res = solve(val, size, count);
        cout<<res<<endl;
    }  
    return 0;  
}  
```

## B

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


void solve(char val[]) {
    int end = strlen(val)-1;
    int i;
    while(end != 0) {
        for(i=0;i<end;i++) {
            if(val[i] > val[i+1]) {
                val[i] -= 1;
                for(int j=i+1;j<=end;j++)
                    val[j] = '9';
                break;
            }
        }
        if(i==end)
            end = 0;
        else
            end = i;
    }
}


int main()  
{  
    freopen("A-large.in", "r", stdin);  
    //freopen("in.txt", "r",stdin);  
    freopen("out.txt", "w", stdout);  
    int t;  
    scanf("%d", &t);  
    for (int i = 1; i<= t; i++)  
    {  
        char val[20] = {0};
        cin>>val;
        printf("Case #%d: ", i);  
        solve(val);
        int k = 0;
        while(val[k] == '0')k++;
        for(;k<strlen(val);k++)
            cout<<val[k];
        cout<<endl;
    }  
    return 0;  
}  
```

## C

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
#include <utility>
using namespace std;

typedef long long ll;


// struct comp{
//     bool operator() (pair<ll, ll> a, pair<ll, ll> b){
//         return (ll)(a.second - a.first) < (ll)(b.second - b.first);
//     }
// };

pair<ll, ll> solve(ll N, ll K) {
    while(K != 1) {
        if(K % 2 == 0) {
            N /= 2;
            K /= 2;
        }
        else {
            N = (N - 1) / 2;
            K /= 2;
        }
    }
    return make_pair(N / 2, (N - 1) / 2);
}


int main()  
{  
    freopen("C-large.in.txt", "r", stdin);  
    //freopen("in.txt", "r",stdin);  
    freopen("out.txt", "w", stdout);  
    int t;  
    scanf("%d", &t);  
    for (int i = 1; i<= t; i++)  
    {  
        ll N, K;
        cin>>N>>K;
        printf("Case #%d: ", i);  
        pair<ll, ll> res = solve(N, K);
        cout<<res.first<<" "<<res.second<<endl;
    }  
    return 0;  
}  
```