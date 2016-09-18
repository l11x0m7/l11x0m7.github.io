--- 
layout: post 
title: 用BFS求解Word Ladder问题(单词接龙问题)
date: 2016-09-18 
categories: blog 
tags: [leetcode, lintcode, bfs] 
description: BFS求解Word Ladder
--- 

# 用BFS求解Word Ladder问题(单词接龙问题)

Word Ladder总共有两题，题目类似，在leetcode和lintcode上都有。我之前尝试使用DFS不太好做，后转而用BFS思路求解。以下是两题和解法。

### 1.Word Ladder

#### 题目

Given two words *(beginWord and endWord)*, and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time
Each intermediate word must exist in the word list
For example,

Given:
beginWord = `"hit"`
endWord = `"cog"`
wordList = `["hot","dot","dog","lot","log"]`
As one shortest transformation is `"hit" -> "hot" -> "dot" -> "dog" -> "cog"`,
return its length `5`.

Note:

* Return 0 if there is no such transformation sequence.
* All words have the same length.
* All words contain only lowercase alphabetic characters.

#### 思路

按照字典从beginword变到endword，且每一步只能改变一个字母，求解最短的步数。

首先考虑层级关系，如从beginword开始为第一层，和beginword相邻（只有一个字母不同，且不包含beginword）的单词为第二层，而和所有beginword相邻集合内单词相邻的所有单词集合为第三层。以此类推。直到在某一层遇到可以转化为endword的单词，则表示该层为最低层。期间可以舍弃被访问过的在字典中的单词。

#### 代码
```cpp
class Solution {
public:
    int ladderLength(string start, string end, unordered_set<string> &dict) {
        if(dict.empty())
            return 0;
        if(start==end)
            return 1;
        map<string, int> m;
        queue<string> q;
        q.push(start);
        m[start] = 1;
        while(!q.empty()){
            string cur = q.front();
            q.pop();
            int cost = m[cur];
            for(int i=0;i<cur.length();i++){
                string tmp = cur;
                for(char c='a';c<='z';c++){
                    tmp[i] = c;
                    if(dict.find(tmp)!=dict.end()){
                        dict.erase(tmp);
                        m[tmp] = cost + 1;
                        q.push(tmp);
                    }
                    if(tmp==end){
                        return cost+1;
                    }
                }
            }
        }
        return 0;
    }
};
```

### 2.Word Ladder II

#### 题目

Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:

1. Only one letter can be changed at a time
2. Each intermediate word must exist in the word list
For example,

Given:
beginWord = `"hit"`
endWord = `"cog"`
wordList = `["hot","dot","dog","lot","log"]`
Return

```
  [
    ["hit","hot","dot","dog","cog"],
    ["hit","hot","lot","log","cog"]
  ]
```
  
Note:

* All words have the same length.
* All words contain only lowercase alphabetic characters.

#### 思路

和上一题差不多，只不过是要把所有的具体结果求出来。只需要按照上面的思路，但是不能在访问一个单词后就删除单词，而是在访问同一层所有单词后，才可以把所有单词删除掉。如果在某一层出现了从begin到end的路径，则该层即为最低层。

#### 代码

```cpp
class Solution {
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, unordered_set<string> &wordList) {
        vector<vector<string>> res;
        queue<vector<string>> path;
        if(wordList.empty())
            return res;
        if(beginWord==endWord){
            string tmp[] = {beginWord, endWord};
            res.push_back(vector<string>(tmp, tmp+1));
            res.push_back(vector<string>(tmp, tmp+2));
            return res;
        }
        unordered_set<string> tag;
        wordList.insert(endWord);
        int min_lev = INT_MAX;
        int level = 1;
        path.push({beginWord});
        tag.insert(beginWord);
        while(!path.empty()){
            vector<string> curway = path.front();
            path.pop();
            if(curway.size()>level){
                for(string tmp:tag)wordList.erase(tmp);
                tag.clear();
                if(curway.size()>min_lev)
                    break;
                else
                    level = curway.size();
            }
            string last = curway.back();
            for(int i=0;i<last.length();i++){
                string tmp = last;
                char curc = last[i];
                for(char c='a';c<='z';c++){
                    if(curc==c)
                        continue;
                    tmp[i] = c;
                    if(wordList.find(tmp)!=wordList.end()){
                        tag.insert(tmp);
                        curway.push_back(tmp);
                        if(tmp==endWord){
                            res.push_back(curway);
                            min_lev = level;
                        }
                        else
                            path.push(curway);
                        curway.pop_back();
                    }
                }
            }
        }
        return res;
    }
};
```