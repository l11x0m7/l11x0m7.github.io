--- 
layout: post 
title: 并查集的简单介绍
date: 2017-04-02 
categories: blog 
tags: [算法, leetcode] 
description: 并查集
--- 

# 并查集的简单介绍

为什么说是简单介绍？嗯，因为没有深入研究，只是最近做题目的时候遇到了，所以写个简洁的介绍，下次需要用的时候，可以直接拿来参考。额……因为很少用到，所以也不知道下一次看这个会是什么时候，哈哈。

并查集（Union Find）中比较经典的问题就是家族问题。假设A和B是亲戚，B和C是亲戚，那么A和C也是亲戚。在图中我们用连通性来表示亲戚关系。那么给定m个人，以及n组二元关系，如何求出到底有多少个家族？

* 按集合方式考虑，A在某几个集合中有亲戚，那么将这几个集合取并集，并加入A，此时的集合为一个家族。如此重复，得到的集合个数就是家族的个数。
* 按森林方式考虑，如果A和B之间是亲戚关系，A和B应该具有相同的家族标志（此处用root值表示），那么将A所在的树连到B上（A所在的树成了B的子树），或相反，则形成了新的家族树。如此反复，则可以构造出家族森林。

并查集有很多种不同的优化形式，常见的有quick-union、quick-find、weighted quick-union、quick-union path compressed。我们现在就quick-union path compressed来进行实现。

quick-union path compressed首先通过quick-union产生几颗树，然后遍历每颗树，将每棵树变为高度为2的树。

union-find是不考虑具体路径的，它只考虑节点的连通性。如果需要得到具体的连通路径，需要使用DFS或者BFS或者别的算法。

下面通过leetcode的两道题来展示如何使用并查集：

#### Friend Circles

地址：https://leetcode.com/contest/leetcode-weekly-contest-26/problems/friend-circles/

union-find的path compressed算法实现如下：

```cpp
class Solution {
public:
    int findCircleNum(vector<vector<int>>& M) {
        int n = M.size();
        vector<int> folk(n, 0);
        if(n == 0)
            return 0;
        for (int i = 0 ; i < n ; i++) folk[i] = i;
        for(int i=0;i<n;i++) {
            for(int j=i+1;j<n;j++) {
                if(M[i][j] != 0) {
                    int _u = dsu(i, folk) , _v = dsu(j, folk);
                    if (_u != _v) {
                        folk[_v] = _u;
                    }
                }
            }
        }
        for(int i = 0 ; i < n ; i++) dsu(i, folk);
        unordered_set<int> s;
        for(auto i : folk) {
            s.insert(i);
            // cout<<i<<endl;
        }
        return s.size();
    }
    int dsu (int u, vector<int>& folk) { 
        return u == folk[u] ? u : folk[u] = dsu(folk[u], folk);
    }
};
```


#### Surrounded Regions

地址：https://leetcode.com/problems/surrounded-regions/#/description

这道题其实只是稍微借用了一下union-find的思路，就是如何得到两个不同的集合。这题的难点在于，在做DFS的时候，会出现递归深度太深，从而造成栈空间不够，产生`Runtime Error`错误。

由于会递归，下面的这个代码的递归深度可能会达到`m*n`（因为会访问到所有点），从而造成runtime error：

```cpp
class Solution {
public:
    void solve(vector<vector<char>>& board) {
        int m = board.size();
        if(m < 1)
            return;
        int n = board[0].size();
        cout<<m<<" "<<n<<endl;
        // unordered_set<string> s;
        vector<vector<bool>> flag(m, vector<bool>(n, false));
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                if(i > 0 && i < m-1 && j > 0 && j < n-1)
                    continue;
                if(board[i][j] == 'O' && !flag[i][j])
                    findRegion(board, flag, i, j);
            }
        }
        
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                if(board[i][j] == 'O' && !flag[i][j])
                    board[i][j] = 'X';
            }
        }
    }
    void findRegion(vector<vector<char>>& board, vector<vector<bool>>& flag, int i, int j) {
        int m = board.size();
        int n = board[0].size();
        flag[i][j] = true;
        if(i > 0 && board[i-1][j] == 'O' && !flag[i-1][j]) findRegion(board, flag, i-1, j);
        if(i < m-1 && board[i+1][j] == 'O' && !flag[i+1][j]) findRegion(board, flag, i+1, j);
        if(j > 0 && board[i][j-1] == 'O' && !flag[i][j-1]) findRegion(board, flag, i, j-1);
        if(j < n-1 && board[i][j+1] == 'O' && !flag[i][j+1]) findRegion(board, flag, i, j+1);
    }
};
```


改进的方法就是每个边缘点只扫描board内部的点，而不会扫描除了该点外的其他边缘的点，因此深度最多就是`max(m, n)`：

```cpp
class Solution {
public:
    void solve(vector<vector<char>>& board) {
        int m = board.size();
        if(m < 1)
            return;
        int n = board[0].size();
        cout<<m<<" "<<n<<endl;
        // unordered_set<string> s;
        vector<vector<bool>> flag(m, vector<bool>(n, false));
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                if(i > 0 && i < m-1 && j > 0 && j < n-1)
                    continue;
                // cout<<i<<" "<<j<<endl;
                if(board[i][j] == 'O' && !flag[i][j])
                    findRegion(board, flag, i, j);
            }
        }
        
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                if(board[i][j] == 'O' && !flag[i][j])
                    board[i][j] = 'X';
            }
        }
    }
    void findRegion(vector<vector<char>>& board, vector<vector<bool>>& flag, int i, int j) {
        int m = board.size();
        int n = board[0].size();
        flag[i][j] = true;
        // cout<<i<<"/"<<j<<endl;
        if(i > 1 && board[i-1][j] == 'O' && !flag[i-1][j]) findRegion(board, flag, i-1, j);
        if(i < m-2 && board[i+1][j] == 'O' && !flag[i+1][j]) findRegion(board, flag, i+1, j);
        if(j > 1 && board[i][j-1] == 'O' && !flag[i][j-1]) findRegion(board, flag, i, j-1);
        if(j < n-2 && board[i][j+1] == 'O' && !flag[i][j+1]) findRegion(board, flag, i, j+1);
    }
};

```