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
```

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

# 466. Count The Repetitions

#### 题目

Define `S = [s,n]` as the string S which consists of n connected strings s. For example, `["abc", 3]` ="abcabcabc".

On the other hand, we define that string s1 can be obtained from string s2 if we can remove some characters from s2 such that it becomes s1. For example, “abc” can be obtained from “abdbec” based on our definition, but it can not be obtained from “acbbe”.

You are given two non-empty strings s1 and s2 (each at most 100 characters long) and two integers 0 ≤ n1 ≤ 106 and 1 ≤ n2 ≤ 106. Now consider the strings S1 and S2, where `S1=[s1,n1]` and `S2=[s2,n2]`. Find the maximum integer M such that `[S2,M]` can be obtained from `S1`.

**Example**  

```
Input:
s1="acb", n1=4
s2="ab", n2=2

Return:
2
```

#### 思路
这道题考虑这样两种情况：

* s2在s1中的出现有一定的循环规律，比如s1=“eacfgeae”，n1=5，s2=“ea”，这样每2个s1可以出现4个s2，且有一个s2被分割了。这里有种特殊情况就是刚好没被分割，但是做法和被分割的一样。
* s2在s1中没有出现规律。

这样，我们只要遍历n1次s1，找到有没有循环规律，如果有，则跳出，没有则执行到结束。

时间复杂度：O(n1*len(s1))

#### 代码

```cpp
class Solution {
public:
    int getMaxRepetitions(string s1, int n1, string s2, int n2) {
        int len1 = s1.size();
        int len2 = s2.size();
        int a[n1]{};
        int b[len2+1]{};
        int j = 0;
        int m = 0;
        int k = 1;
        for(;k<=n1;k++){
            for(auto c:s1){
                if(c==s2[j])j++;
                if(j==len2){
                    m++;
                    j = 0;
                }
            }
            a[k] = m;
            if(!j||b[j])break;
            b[j] = k;
        }
        if(n1==k)return a[n1]/n2;
        n1 -= b[j];
        return (n1 / (k - b[j]) * (a[k] - a[b[j]]) + a[n1 % (k - b[j]) + b[j]]) / n2;
    }
};
```

# 472. Concatenated Words

#### 题目

Given a list of words (without duplicates), please write a program that returns all concatenated words in the given list of words.

A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.

**Example:**

```
Input: ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]

Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]

Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats"; 
 "dogcatsdog" can be concatenated by "dog", "cats" and "dog"; 
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".
```

**Note:**  

1. The number of elements of the given array will not exceed **10,000**.
2. The length sum of elements in the given array will not exceed **600,000**.
3. All the input string will only include lower case letters.
4. The returned elements order does not matter.

#### 思路1

Trie思路：  

* 先对所有词建立一棵Trie树，把所有单词路径都存进去。
* 用dfs遍历每个词，看这个词能否被拆成两个或两个以上的词。这里要注意，最后一个Test会TLE，所以可以用静态空间代替堆空间来建立Trie树，以减少内存使用。

#### 代码1

```cpp
struct Node{
    bool isword;
    Node* child[26];
    Node(){
        isword = false;
        memset(child, NULL, sizeof(Node*)*26);
    }
};
class Solution {
public:
    vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
        vector<string> res;
        static Node node_pool[60000], *root = node_pool;
        memset(node_pool, 0, sizeof(node_pool));
        int count = 1;
        Node* r;
        for(auto i:words){
            r = root;
            for(int j=0;j<i.size();j++){
                if(r->child[i[j]-'a']==nullptr)
                    r->child[i[j]-'a'] = &node_pool[count++];
                r = r->child[i[j]-'a'];
            }
            r->isword = true;
        }
        for(auto word:words){
            if(dfs(word, root, 0, 0))
                res.push_back(word);
        }
        return res;
    }
    bool dfs(string& word, Node* r, int ind, int count){
        if(ind>=word.size()){
            if(count>=2)
                return true;
            return false;
        }
        Node* root = r;
        int i = ind;
        while(i<word.size()&&r->child[word[i]-'a']!=nullptr){
            r = r->child[word[i]-'a'];
            i++;
            if(r->isword&&dfs(word, root, i, count+1)){
                return true;
            }
        }
        return false;
    }
};
```

#### 思路2

DP思路：

* 对每个词先进行排序，短的放前面，长的放后面。
* 对每个词进行遍历，看这个词能否被拆成多个词，如果可以，则存入res，如果不能，则作为单词存入set

#### 代码2

```cpp
class Solution {
public:
    vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
        vector<string> res;
        unordered_set<string> s;
        auto comp = [](string a, string b){return a.size()<b.size();};
        sort(words.begin(), words.end(), comp);
        for(auto word:words){
            if(dfs(word, s))
                res.push_back(word);
            else
                s.insert(word);
        }
        return res;
    }
    bool dfs(string& word, unordered_set<string>& s){
        if(word=="")
            return false;
        vector<bool> dp(word.size()+1, false);
        dp[0] = true;
        for(int i=1;i<=word.size();i++){
            for(int j=i-1;j>=0;j--){
                if(dp[j]&&s.find(word.substr(j, i-j))!=s.end()){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[word.size()];
    }
};
```