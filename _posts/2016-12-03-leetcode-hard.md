--- 
layout: post 
title: Leetcode的Hard难度题目汇总
date: 2016-12-03 
categories: blog 
tags: [算法, leetcode] 
description: 
--- 

# 126. Word Ladder II

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

使用广度优先遍历，查询每个从begin开始的所有邻近单词。判定在该层上是否有到达end的单词，如果有，则保存该层上所有的能够到达终点的路径；如果没有，则保存当前所有路径，并将这些词从原始字典中删除，开始下一层的遍历，直到在某一层找到能够到达end的路径。

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


# 149. Max Points on a Line

#### 题目

`Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.`

#### 思路  
算法的基本思路是遍历每个点，找到每个点对应的线段斜率，并统计斜率相同的线段数，从而得到结果。这里注意两个特殊情况：

* 两点在同一垂直线上
* 两点坐标重叠

复杂度：O(n^2)

注意如果使用的是map而不是unordered_map，则是O(n^2logn)。

#### 代码

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


# 146. LRU Cache

#### 题目

Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: `get` and `set`.

`get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
`set(key, value)` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

#### 思路

第一种，使用带空head和空rear的链表，每次get的时候将被get的值移动到前面（移动的时间复杂度为O(1)，但是查询的时间复杂度为O(n)）；每次set的时候，
* 如果该值在原列表中存在，则将该值移动到开头；
* 如果不在原链表中，且没有达到容量，则插入到链表开头；
* 如果不在原链表中，且达到了容量，则删除链表尾元素，并插入到链表开头。

每次查找链表中对应key的时候都要遍历链表，因此复杂度较低。  
第二种，可以考虑用map做这样的映射：key->pair<value, 在链表中的位置指针>。

#### 代码

```cpp
# 第一种


```
struct Node{
    int key;
    int value;
    Node* next;
    Node* pre;
    Node(int key=-1, int value=-1){
        this->key = key;
        this->value = value;
        next=NULL,pre=NULL;
    }
};
class LRUCache{
public:
    int cap;
    int len;
    unordered_map<int, int> m;
    Node* head;
    Node* rear;
    // @param capacity, an integer
    LRUCache(int capacity) {
        // write your code here
        cap = capacity;
        head = new Node();
        rear = new Node();
        head->next = rear;
        rear->pre = head;
        len = 0;
    }
    
    // @return an integer
    int get(int key) {
        // write your code here
        if(m.find(key)!=m.end()){
            Node* h = head;
            while(h&&h->key!=key)
                h = h->next;
            moveFront(h);
            return m[key];
        }
        else
            return -1;
    }

    // @param key, an integer
    // @param value, an integer
    // @return nothing
    void set(int key, int value) {
        // write your code here
        if(cap<1)
            return;
        if(m.find(key)==m.end()){
            m[key] = value;
            if(len>=cap)
                deleteLast();
            else
                len++;
        
            Node* h = new Node(key, value);
            h->next = head->next;
            head->next->pre = h;
            head->next = h;
            h->pre = head;
        }
        else{
            Node* h = head;
            while(h&&h->key!=key)
                h = h->next;
            h->value = value;
            m[key] = value;
            moveFront(h);
        }
        
    }
    void moveFront(Node* &h){
        h->pre->next = h->next;
        h->next->pre = h->pre;
        h->next = head->next;
        head->next->pre = h;
        h->pre = head;
        head->next = h;
    }
    void deleteLast(){
        Node* h = head;
        while(h->next->next){
            h = h->next;
        }
        h->pre->next = h->next;
        h->next->pre = h->pre;
        m.erase(h->key);
        // delete h;
    }
};
```cpp
# 第二种
class LRUCache{
public:
    int cap;
    list<int> l;
    unordered_map<int, pair<int, list<int>::iterator>> m;
    // @param capacity, an integer
    LRUCache(int capacity) {
        // write your code here
        cap = capacity;
    }
    
    // @return an integer
    int get(int key) {
        // write your code here
        auto it = m.find(key);
        if(it!=m.end()){
            update(it);
            return it->second.first;
        }
        else
            return -1;
    }

    // @param key, an integer
    // @param value, an integer
    // @return nothing
    void set(int key, int value) {
        // write your code here
        if(cap<1)
            return;
        auto it = m.find(key);
        if(it==m.end()){
            if(m.size()>=cap){
                m.erase(l.back());
                l.pop_back();
            }
            l.push_front(key);
        }
        else
            update(it);
        m[key] = make_pair(value, l.begin());
    }
    void update(unordered_map<int, pair<int, list<int>::iterator>>::iterator it){
        int key = it->first;
        l.erase(it->second.second);
        l.push_front(key);
        it->second.second = l.begin();
    }
};
```

# 460. LFU Cache

#### 题目

Design and implement a data structure for Least Frequently Used (LFU) cache. It should support the following operations: `get` and `set`.

* `get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
* `set(key, value)` - Set or insert the value if the key is not already present. When the cache reaches its capacity, it should invalidate the least frequently used item before inserting a new item. For the purpose of this problem, when there is a tie (i.e., two or more keys that have the same frequency), the least recently used key would be evicted.

Follow up:  
Could you do both operations in **O(1)** time complexity?

Example:

```
LFUCache cache = new LFUCache( 2 /* capacity */ );

cache.set(1, 1);
cache.set(2, 2);
cache.get(1);       // returns 1
cache.set(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.get(3);       // returns 3.
cache.set(4, 4);    // evicts key 1.
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

#### 思路

用三个hash表，分别为：

* key -> (freq, list iterator)
* key -> value
* freq -> list

相当于将每个键按照出现的不同频率进行分桶。每个频率对应一个存储key的列表。而每个key对应所在频率的列表位置，是一个pair。那么在get的时候，我们只需要找到对应key的freq，之后找到该freq下该key的位置即可。在set的时候，如果超出容量，则删除least元素（频率最低、被访问的时间间隔最久），之后再插入新的元素到freq=1的list里。

注意，只要产生可能使列表为空的操作（比如插入或者更新，插入会让最小频率变为1），都要更新最小频率参数least，这个参数用来在删除元素的时候使用。

时间复杂度：O(1)

#### 代码

```cpp
class LFUCache {
public:
    LFUCache(int capacity) {
        this->capacity = capacity;
        least = 1;
    }
    
    int get(int key) {
        auto it = key2pair.find(key);
        int ret;
        if(it!=key2pair.end()){
            update(it);
            ret = key2value[key];
            if(freq2list[least].size()==0){
                least = key2pair[key].first;
                // cout<<"least:"<<least<<endl;
            }
        }
        else
            ret = -1;
        return ret;
    }
    
    void set(int key, int value) {
        if(capacity<=0)
            return;
        int val = get(key);
        if(val!=-1){
            key2value[key] = value;
        }
        else{
            if(key2pair.size()>=capacity){
                key2pair.erase(freq2list[least].back());
                key2value.erase(freq2list[least].back());
                freq2list[least].pop_back();
            }
            key2value[key] = value;
            freq2list[1].push_front(key);
            key2pair[key].first = 1;
            key2pair[key].second = freq2list[1].begin();
            least = 1;
        }
        // for(auto i:freq2list){
        //     cout<<(i.first)<<" ";
        //     for(auto j:(i.second))
        //         cout<<j<<" ";
        //     cout<<endl;
        // }
        
    }
    void update(unordered_map<int, pair<int, list<int>::iterator>>::iterator it){
            int freq = it->second.first;
            key2pair[it->first].first = freq + 1;
            freq2list[freq].erase(it->second.second);
            // cout<<"freq:"<<freq<<" "<<freq2list[freq].size()<<endl;
            freq2list[freq+1].push_front(it->first);
            it->second.second = freq2list[freq+1].begin();
    }
private:
    int capacity;
    int least;
    unordered_map<int, pair<int, list<int>::iterator>> key2pair;
    unordered_map<int, int> key2value;
    unordered_map<int, list<int>> freq2list;
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache obj = new LFUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.set(key,value);
 */
```