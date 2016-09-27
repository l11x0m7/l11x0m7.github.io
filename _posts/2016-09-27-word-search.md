--- 
layout: post 
title: 词搜索与Trie树
date: 2016-09-27 
categories: blog 
tags: [算法, lintcode, leetcode] 
description: 两个词搜索的例子与Trie树
--- 

# 词搜索与Trie树

### Trie树

Trie树是一种比较简单的字典树，它通过前缀来构造单词的路径。当搜索一个单词的时候可以从根节点出发，一直找到单词节点（可以使用一个变量来标定）。如果该单词能够找到单词节点，那么说明这个词在字典里，否则不在（单词未搜索完或者搜索完但是未达到单词节点）。

如果字典里的字都是由小写字母构成（26个），那么可以使用如下数据结构：

```cpp
struct TreeNode{
        bool isword;
        string word;
        TreeNode* next[26];
        TreeNode(){
            isword = false;
        }
    }
```

其中，每个变量的含义如下：

* isword：判定是否为一个单词节点（即单词搜索到此处是否可以形成一个单词）;
* word：如果是一个单词节点，则把该单词存入到word里;
* next：表示下一个字母，共26种可能，'a'到'z'.

### 词搜索

给定一个单词，如何在一个字母矩阵中找到这个单词？

如果仅仅是要判定这个单词是否在字母矩阵中，那么仅使用dfs即可。时间复杂度为O(N^2)。

如果额外给定了字典，查找所有字母矩阵中出现过的字典里的词，那么需要对**字典**建立**Trie树**，然后使用dfs搜索即可。时间复杂度为O(logN*N^2)。



#### 词搜索I

```
给出一个二维的字母板和一个单词，寻找字母板网格中是否存在这个单词。

单词可以由按顺序的相邻单元的字母组成，其中相邻单元指的是水平或者垂直方向相邻。每个单元中的字母最多只能使用一次。


给出board =

[

  "ABCE",

  "SFCS",

  "ADEE"

]

word = "ABCCED"， ->返回 true,

word = "SEE"，-> 返回 true,

word = "ABCB"， -> 返回 false.
```

我的代码如下：

```cpp
class Solution {
public:
    /**
     * @param board: A list of lists of character
     * @param word: A string
     * @return: A boolean
     */
    bool exist(vector<vector<char> > &board, string word) {
        // write your code here
        int n = board.size();
        if(word.empty())
            return true;
        if(n<1)
            return false;
        int m = board[0].size();
        vector<vector<bool>> isvisit(n, vector<bool>(m, false));
        for(int i=0;i<board.size();i++){
            for(int j=0;j<board[0].size();j++){
                if(board[i][j]!=word[0])
                    continue;
                if(dfs(board, isvisit, word, 0, i, j))
                    return true;
            }
        }
        return false;
    }
    bool dfs(vector<vector<char>>& board, vector<vector<bool>>& isvisit, string& word, int k, int i, int j){
        if(k==word.size()-1&&word[k]==board[i][j])
            return true;
        isvisit[i][j] = true;
        if(word[k]==board[i][j]){
            // cout<<word[k]<<endl;
            if(i>0&&!isvisit[i-1][j])
                if(dfs(board, isvisit, word, k+1, i-1, j))
                    return true;
            if(i<board.size()-1&&!isvisit[i+1][j])
                if(dfs(board, isvisit, word, k+1, i+1, j))
                    return true;
            if(j>0&&!isvisit[i][j-1])
                if(dfs(board, isvisit, word, k+1, i, j-1))
                    return true;
            if(j<board[0].size()-1&&!isvisit[i][j+1])
                if(dfs(board, isvisit, word, k+1, i, j+1))
                    return true;
        }
        isvisit[i][j] = false;
        return false;
    }
};
```

#### 词搜索II

```
给出一个由小写字母组成的矩阵和一个字典。找出所有同时在字典和矩阵中出现的单词。一个单词可以从矩阵中的任意位置开始，可以向左/右/上/下四个相邻方向移动。
```

```
样例
给出矩阵：
doaf
agai
dcan
和字典：
{"dog", "dad", "dgdg", "can", "again"}

返回 {"dog", "dad", "can", "again"}
```

dog:
**do**af
a**g**ai
dcan
dad:
**d**oaf
**a**gai
**d**can
can:
doaf
agai
d**can**
again:
doaf
**agai**
dca**n**


我的代码如下：

```cpp
class Solution {
public:
    /**
     * @param board: A list of lists of character
     * @param words: A list of string
     * @return: A list of string
     */
    struct TreeNode{
        bool isword;
        bool isfind;	# 这个是另外加的，用来标注该单词是否被搜索过，防止结果有重复
        string word;
        TreeNode* next[26];
        TreeNode(){
            isword = false;
            isfind = false;
        }
    } *root;
    vector<string> wordSearchII(vector<vector<char> > &board, vector<string> &words) {
        // write your code here
        root = new TreeNode();
        for(auto word:words)
            buildTrie(word, root);
        vector<string> res;
        vector<vector<bool>> isvisit(board.size(), vector<bool>(board[0].size(), false));
        for(int i=0;i<board.size();i++)
            for(int j=0;j<board[0].size();j++)
                dfs(board, i, j, root, res, isvisit);
        return res;
    }
    # 构造Trie树
    void buildTrie(string& word, TreeNode* r){
        for(auto c:word){
            if(!r->next[c-'a'])
                r->next[c-'a'] = new TreeNode();
            r = r->next[c-'a'];
        }
        r->isword = true;
        r->word = word;
    }
    void dfs(vector<vector<char>>& board, int i, int j, TreeNode* r, vector<string>& res, vector<vector<bool>>& isvisit){
        char c = board[i][j];
        isvisit[i][j] = true;
        // cout<<c<<endl;
        if(r->next[c-'a']){
            r = r->next[c-'a'];
            if(r->isword&&!r->isfind){
                res.push_back(r->word);
                r->isfind=true;
            }
            if(i>0&&!isvisit[i-1][j])
                dfs(board, i-1, j, r, res, isvisit);
            if(i<board.size()-1&&!isvisit[i+1][j])
                dfs(board, i+1, j, r, res, isvisit);
            if(j>0&&!isvisit[i][j-1])
                dfs(board, i, j-1, r, res, isvisit);
            if(j<board[0].size()-1&&!isvisit[i][j+1])
                dfs(board, i, j+1, r, res, isvisit);
        }
        isvisit[i][j] = false;
    }
};
```