--- 
layout: post 
title: Leetcode的Hard难度题目汇总（持续更新）
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

# 517. Super Washing Machines

#### 题目

You have n super washing machines on a line. Initially, each washing machine has some dresses or is empty.

For each move, you could choose any m (1 ≤ m ≤ n) washing machines, and pass one dress of each washing machine to one of its adjacent washing machines at the same time .

Given an integer array representing the number of dresses in each washing machine from left to right on the line, you should find the minimum number of moves to make all the washing machines have the same number of dresses. If it is not possible to do it, return -1.

#### 思路

把所有机器的move最小和转化成每台机器的必需move的最大值（保证每台机器在调整平衡的时候被尽可能多的使用，表示不断重复利用），那么这些机器的最大值中的最大值就是所有机器的move最小值。

对于一台机器来说，计算它左边的机器缺多少（或多多少）件衣服l，右边同样记为r。之后要让左右两边的数平衡，需要考虑l和r的取值：

1. `l >= 0 && r >= 0` 则表示左右都缺衣服，那么需要当前机器输送衣服给左右两边的机器。当前机器（这种情况只有当前机器需要操作）需要操作l+r次；
2. `l < 0 && r >= 0` 则表示左边多，右边少，需要操作`max(l, r)`次；
3. `l >= 0 && r < 0` 则表示左边少，右边多，需要操作`max(l, r)`次；
4. `l < 0 && r < 0`表示两边都多衣服，则需要操作`max(l, r)`次，即把多余的衣服给当前机器。

#### 代码

```cpp
class Solution {
public:
    int findMinMoves(vector<int>& machines) {
        int len = machines.size();
        int sum[len+1];
        sum[0] = 0;
        for(int i=0;i<len;i++)
            sum[i+1] = machines[i] + sum[i];
        if(sum[len] % len != 0) return -1;
        int avg = sum[len] / len;
        int l, r;
        int res = 0;
        for(int i=0;i<len;i++) {
            l = i * avg - sum[i];
            r = (len-i-1) * avg - (sum[len] - sum[i+1]);
            if(l >= 0 && r >= 0)
                res = max(res, l + r);
            else
                res = max(res, max(abs(l), abs(r)));
            // cout<<res<<endl;
        }
        return res;
    }
};
```



# 502. IPO

#### 题目

Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital before the IPO. Since it has limited resources, it can only finish at most k distinct projects before the IPO. Help LeetCode design the best way to maximize its total capital after finishing at most k distinct projects.

You are given several projects. For each project i, it has a pure profit Pi and a minimum capital of Ci is needed to start the corresponding project. Initially, you have W capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.

To sum up, pick a list of at most k distinct projects from given projects to maximize your final capital, and output your final maximized capital.


**Example 1:**

```
Input: k=2, W=0, Profits=[1,2,3], Capital=[0,1,1].

Output: 4

Explanation: Since your initial capital is 0, you can only start the project indexed 0.
After finishing it you will obtain profit 1 and your capital becomes 1.
With capital 1, you can either start the project indexed 1 or the project indexed 2.
Since you can choose at most 2 projects, you need to finish the project indexed 2 to get the maximum capital.
Therefore, output the final maximized capital, which is 0 + 1 + 3 = 4.
```

Note:

1. You may assume all numbers in the input are non-negative integers.
2. The length of Profits array and Capital array will not exceed 50,000.
3. The answer is guaranteed to fit in a 32-bit signed integer.

#### 思路

此题的思路就是贪婪算法+优先级队列。

一开始我有W块钱，只能买W以及以下的东西，但是买了以后不花钱（因为是pure profit），且可以净赚`Profit[i]`块，但只能买k个，且不重样。那么一个简单的思路就是我每次买的时候，先扫描我能买的，然后选择可以获得的净利润`Profit[i]`，循环k遍，就得到最大的利润。

但是上述做法的复杂度为`O(n*k)`，是暴力解法。其实可以把一开始的数据分在两个容器里，用优先级队列存放我能够的买的东西的`Profit[i]`，另一个用按`Capital`排序的vector存放我暂时买不了的东西的`pair(Capital[i], Profit[i])`。这样我每次买的时候，从优先级队列中取出最大利益值，并更新W，之后再根据更新过的W从vector中选择我现在可以买的东西的`Profit[i]`到优先级队列中。

#### 代码

```cpp
class Solution {
public:
    int findMaximizedCapital(int k, int W, vector<int>& Profits, vector<int>& Capital) {
        int len = Profits.size();
        priority_queue<int> pq;
        vector<pair<int, int>> cp;
        for(int i=0;i<len;i++) {
            if(Profits[i]>0) {
                if(W>=Capital[i])
                    pq.push(Profits[i]);
                else
                    cp.push_back(pair<int, int>(Capital[i], Profits[i]));
            }
        }
        auto f = [](pair<int, int> a, pair<int, int> b) {return a.first<b.first;};
        sort(cp.begin(), cp.end(), f);
        while(k--) {
            if(pq.empty())
                return W;
            W += pq.top();
            pq.pop();
            int i = 0;
            while(!cp.empty()&&cp[i].first<=W) {
                pq.push(cp[i].second);
                cp.erase(cp.begin());
            }
        }
        return W;
    }
};
```


# 514. Freedom Trail

#### 题目

[Freedom Trail](https://leetcode.com/problems/freedom-trail/?tab=Description)


#### 思路

比较正式的思路就是DP，每次我们考虑两个多源问题：

1. 从上面的哪个位置跳到当前匹配的位置（对应多个来源）；
2. 从上面的位置跳到匹配key中当前字符的ring所在的位置（对应多个目的）。

其实就是从ring中的哪个上一目标位置跳到ring中的哪个当前目标位置。

这样，我们可以用`dp[i][j]`来存储目标`key[i]`对应的跳到`ring[j]`的步数，可见`key[i] == ring[j]`，而从多个源跳入当前位置`j`，则需要遍历多个源，然后选择`dp[i][j]`为跳入`j`的步数最小值。

当然也可以用DFS方法来遍历每一种可能。


#### DP代码
```cpp
class Solution {
public:
    int findRotateSteps(string ring, string key) {
        int m = ring.size();
        int n = key.size();
        if(m < 1 || n < 1)
            return 0;
        int dp[n+1][m];
        memset(dp, 0, sizeof(dp));
        set<int> pre;
        pre.insert(0);
        for(int i=0;i<n;i++) {
            set<int> tmp;
            for(int j=0;j<m;j++) {
                if(key[i] == ring[j]) {
                    // cout<<j<<ring[j]<<endl;
                    for(auto k : pre) {
                        int step = abs(j - k);
                        step = min(step, m - step);
                        int jump = dp[i][k] + step + 1;
                        if(dp[i+1][j] != 0)
                            dp[i+1][j] = min(dp[i+1][j], jump);
                        else
                            dp[i+1][j] = jump;
                        // cout<<dp[i+1][j]<<endl;
                    }
                    tmp.insert(j);
                }
            }
            pre = tmp;
        }
        int res = INT_MAX;
        for(int i=0;i<m;i++) {
            // cout<<dp[n][i];
            if(ring[i] == key[n-1])
                res = min(res, dp[n][i]);
        }
        return res;
    }
};
```

# 546. Remove Boxes

#### 题目

Given several boxes with different colors represented by different positive numbers. 
You may experience several rounds to remove boxes until there is no box left. Each time you can choose some continuous boxes with the same color (composed of k boxes, `k >= 1`), remove them and get `k*k` points.
Find the maximum points you can get.


**Example 1:**

Input:

```
[1, 3, 2, 2, 2, 3, 4, 3, 1]
```

Output:

```
23
```

Explanation:  

```
[1, 3, 2, 2, 2, 3, 4, 3, 1] 
----> [1, 3, 3, 4, 3, 1] (3*3=9 points) 
----> [1, 3, 3, 3, 1] (1*1=1 points) 
----> [1, 1] (3*3=9 points) 
----> [] (2*2=4 points)
```

**Note:** The number of boxes `n` would not exceed 100.

#### 思路

这题的思路参考了[Python, Fast DP with Explanation](https://discuss.leetcode.com/topic/84307/python-fast-dp-with-explanation)

使用DP+DFS来完成。`dp[i][j][k]`表示从i和j之间的子数组在i之前共有k个值等于`boxes[i]`的条件下的最大得分数。这样，每个`dp[i][j][k]`的值有两种方式：

* 如果从i到m（`i<m<=j`）的数都等于`boxes[i]`，那么我们就可以考虑让前面的`k`个和这`m-i`个数合并，那么和为`helper(boxes, dp, m+1, j, 0) + (k+1) * (k+1)`
* 如果从m+1到j当中存在某个位置l，使得`boxes[l] == boxes[m]`，那么就把该数前面的`m+1`到`l-1`的数合并，为`helper(boxes, dp, m+1, l-1, 0)`；并加上从`l`到`j`的合并的所有数，为`helper(boxes, dp, l, j, k+1)`

其实第一种情况也可以拆解成第二种情况，按多步完成，但是由于多个数连在一起进行消除的分数一定比单个拆开来要高，所以可以单独做这一步。



#### 代码

```cpp
class Solution {
public:
    int removeBoxes(vector<int>& boxes) {
        int n = boxes.size();
        int dp[100][100][100] = {0};
        return helper(boxes, dp, 0, n-1, 0);
    }
    int helper(vector<int>& boxes, int dp[][100][100], int i, int j, int k) {
        if(i>j)return 0;
        if(dp[i][j][k] != 0)
            return dp[i][j][k];
        int m = i;
        while(m+1<=j && boxes[i] == boxes[m+1]) m++;
        k = k + (m-i);
        int res = helper(boxes, dp, m+1, j, 0) + (k+1) * (k+1);
        for(int l=m+1;l<=j;l++) {
            if(boxes[l] == boxes[m]) {
                res = max(res, helper(boxes, dp, m+1, l-1, 0) + helper(boxes, dp, l, j, k+1));
            }
        }
        dp[m][j][k] = res;
        return res;
    }
};
```


# 564. Find the Closest Palindrome

#### 题目

Given an integer n, find the closest integer (not including itself), which is a palindrome.

The 'closest' is defined as absolute difference minimized between two integers.

**Example 1:**

```
Input: "123"
Output: "121"
```

**Note:**

1. The input n is a positive integer represented by string, whose length will not exceed 18.
2. If there is a tie, return the smaller one as answer.


#### 思路

首先，对于一个数，只考虑它的前一半。比如`12367`，那么它的前一半就是`123`。现在对该前半部分进行调整，那么离它最近的回文的前缀一定在`[12221,12321,12421]`当中。考虑另一种情况，如果是边缘值，比如`1000`，`9999`，那么离他们最近的就是[999,1001]中的一个，或者`[10001,9999]`中的一个。所以，先得到这些所有的候选词，然后选出最近的或者题目最优的。步骤如下：

1. 对`s`，求出`s`的前缀部分`head`;
2. 对`head`求`[head-1,head,head+1]`对应的回文；
3. 对`s`求相应位数与高一位位数的边界值（如果s为4位，则求出1001，999，9999，10001）；
4. 对以上的候选值进行筛选。

> 注意，本题使用long long即可，因为long long最长表示比10^18大。

#### 代码

```cpp
class Solution {
public:
    string nearestPalindromic(string n) {
        int len = n.size();
        vector<string> candidates;
        for(int i=len-1;i<=len;i++) {
            for(int j=-1;j<=1;j+=2) {
                candidates.push_back(to_string((long long)pow(10, i) + (long long)j));
            }
        }
        string pre = n.substr(0, (len+1)/2);
        // cout<<pre<<endl;
        for(int i=-1;i<=1;i++) {
            string cur = to_string(stoll(pre) + (long long)i);
            string lat = cur;
            reverse(lat.begin(), lat.end());
            if(len % 2 == 1)
                cur = cur.substr(0, cur.size()-1) + lat;
            else
                cur += lat;
            candidates.push_back(cur);
        }
        string res = "";
        for(string s: candidates) {
            if(s != n && (res == "" || abs(stoll(res) - stoll(n))>abs(stoll(s) - stoll(n)) || (abs(stoll(res) - stoll(n))==abs(stoll(s) - stoll(n)) && stoll(s) < stoll(res))))
                res = s;
        }
        return res;
    }
};
```


# 552. Student Attendance Record II

#### 题目

Given a positive integer n, return the number of all possible attendance records with length n, which will be regarded as rewardable. The answer may be very large, return it after mod $10^9$ + 7.

A student attendance record is a string that only contains the following three characters:

1. 'A' : Absent.
2. 'L' : Late.
3. 'P' : Present.

A record is regarded as rewardable if it doesn't contain more than one 'A' (absent) or more than two continuous 'L' (late).

**Example 1:**

```
Input: n = 2
Output: 8 
Explanation:
There are 8 records with length 2 will be regarded as rewardable:
"PP" , "AP", "PA", "LP", "PL", "AL", "LA", "LL"
Only "AA" won't be regarded as rewardable owing to more than one absent times. 
```

**Note:** The value of n won't exceed 100,000.


#### 思路

使用`dp[i][j]`。假设i表示对应的第i个数，j只有3个值`0,1,2`对应`A,L,P`，dp表示在当前位置取三个值对应的符合要求的情况。考虑三种情况：

1. 当前位i放`P`时，前面的可以随意放，就是`dp[i][0]+dp[i][1]+dp[i][2]`
2. 当前位i放`L`时，前面一位可以是`A`或`P`，或者放`L`且`i-2`位置不放`L`
3. 当前位i放`A`时，前面所有的不能出现`A`，那么就是`dp[i-1][0] + dp[i-2][0] + dp[i-3][0]`，第一个表示`i-1`放`P`且前面无`A`，第二个表示`i-1`放`L`且`i-2`放`P`且前面无`A`，第三个表示`i-1`放`L`且`i-2`放`L`且`i-3`放`P`且前面无`A`的个数。

#### 代码

MLE的代码：


```cpp
class Solution {
public:
    int checkRecord(int n) {
        if(n < 1)
            return 0;
        if(n == 1)
            return 3;
        vector<vector<long long>> dp(n+1, vector<long long>(3, 0));
        // A L P
        dp[0][0] = 1;
        dp[1][0] = 1;
        dp[1][1] = 1;
        dp[1][2] = 1;
        dp[2][0] = 2;
        dp[2][1] = 3;
        dp[2][2] = 3;
        for(int i=3;i<=n;i++) {
            dp[i][2] = (dp[i-1][0] + dp[i-1][1] + dp[i-1][2]) % 1000000007;
            dp[i][1] = ((dp[i-2][0] + dp[i-2][2]) + dp[i-1][0] + dp[i-1][2]) % 1000000007;
            dp[i][0] = (dp[i-1][0] + dp[i-2][0] + dp[i-3][0]) % 1000000007;
        }
        return (dp[n][0] + dp[n][1] + dp[n][2]) % 1000000007;
    }
};
```

修改后的代码：

```cpp
class Solution {
public:
    int checkRecord(int n) {
        if(n < 1)
            return 0;
        if(n == 1)
            return 3;
        if(n == 2)
            return 8;
        long long dp00 = 1;
        long long dp10 = 1;
        long long dp11 = 1;
        long long dp12 = 1;
        long long dp20 = 2;
        long long dp21 = 3;
        long long dp22 = 3;
        long long dp30 = 0;
        long long dp31 = 0;
        long long dp32 = 0;
        for(int i=3;i<=n;i++) {
            dp32 = (dp20 + dp21 + dp22) % 1000000007;
            dp31 = (dp10 + dp12 + dp20 + dp22) % 1000000007;
            dp30 = (dp20 + dp10 + dp00) % 1000000007;
            dp00 = dp10;
            dp12 = dp22;
            dp11 = dp21;
            dp10 = dp20;
            dp22 = dp32;
            dp21 = dp31;
            dp20 = dp30;
        }
        return (dp30 + dp31 + dp32) % 1000000007;
    }
};
```

# 493. Reverse Pairs

#### 题目

https://leetcode.com/problems/reverse-pairs/#/description

#### 思路

有三种思路：merge、BST、BIT

可参考下面的链接（general problems）：

https://discuss.leetcode.com/topic/79227/general-principles-behind-problems-similar-to-reverse-pairs/6

#### 代码

按照merge sort的思路：


```cpp
class Solution {
public:
    int reversePairs(vector<int>& nums) {
        int res = merge(nums.begin(), nums.end());
        // for(int i=0;i<nums.size();i++)
        //     cout<<nums[i]<<endl;
        return res;
    }
    int merge(vector<int>::iterator l, vector<int>::iterator r) {
        if(r - l <= 1)
            return 0;
        auto mid = l + (r - l - 1) / 2;
        int res = merge(l, mid + 1) + merge(mid + 1, r);
        auto tmp = mid;
        auto tmp2 = r - 1;
        while(tmp != l-1 && tmp2 != mid) {
            if(*tmp > 2 * (long long)*tmp2) {
                res += (tmp2 - mid);
                tmp--;
            }
            else
                tmp2--;
        }
        inplace_merge(l, mid + 1, r);
        return res;
    }
};
```

# 483. Smallest Good Base

#### 题目

https://leetcode.com/problems/smallest-good-base/#/description

#### 思路

这道题不难，主要是二分搜索。但是难在double（在使用pow的时候的return值）和unsigned long long之间的转化。因为在值很大的情况下，double转ull会有位丢失，需要注意。

二分搜索：

* 给定数n，确定其最多能用几个1表示，那么肯定是在二进制的情况下，1最多；而最少可以用1个1表示，那么base就等于n。得到可表示n的1的个数范围[2, d]；
* 降序遍历[2, d]，然后考虑在k个1的条件下，二分搜索base的值（此时base值范围在[2, pow(n, 1/(k-1))+1]内），使得base满足pow(base, k-1)+...+pow(base, 0)=n，则返回base；
* 如果都找不到，则返回n本身。

#### 代码

```cpp
typedef unsigned long long ll;
class Solution {
public:
    string smallestGoodBase(string n) {
        ll m = (ll)stoll(n);
        int count = 2;
        while((1LL<<count) < m) count++;
        while(count-- > 1) {
            ll lower = 2;
            ll upper = count == 1 ? m : (ll)pow(m, 1./count) + 1;
            while(lower <= upper) {
                ll mid = (lower + upper) / 2;
                ll c = cal(mid, count + 1);
                if(c < m)
                    lower = mid + 1;
                else if(c > m)
                    upper = mid - 1;
                else
                    return to_string(mid);
            }
        }
        return n;
    }
    ll cal(ll mid, int k) {
        ll res = 0;
        ll count = 1;
        while(k--) {
            res += count;
            count *= mid;
        }
        return res;
    }
};
```

# 132. Palindrome Partitioning II

#### 题目

https://leetcode.com/problems/palindrome-partitioning-ii/#/description

#### 思路

对于每个子串对于子串`s[0...j]`，它被cut的最小次数为：`min(cut(s[0...j]), cut(s[0...i-1]) + 1)`，对于所有为palindrome串的`s[i...j]`

#### 代码

```cpp
class Solution {
public:
    int minCut(string s) {
        int N = s.size();
        int mincut[N + 1];
        vector<vector<bool>> isPalin(N, vector<bool>(N, false));
        for(int i=0;i<N;i++) isPalin[i][i] = true;
        for(int i=0;i<=N;i++)mincut[i] = i - 1;
        for(int i=0;i<N;i++) {
            for(int j=i;j>=0;j--) {
                if(s[j] == s[i]) {
                    if(i - j >= 2 && !isPalin[j+1][i-1])
                        continue;
                    mincut[i + 1] = min(mincut[i + 1], mincut[j] + 1);
                    isPalin[j][i] = true;
                }
            }
        }
        return mincut[N];
    }
};
```

# 600. Non-negative Integers without Consecutive Ones

#### 题目

https://leetcode.com/problems/non-negative-integers-without-consecutive-ones/#/solutions


#### 思路

先不考虑这题，我们先假设，给定比特数n，求该比特数对应的`Non-negative Integers without Consecutive Ones`数量(设为f)。  

* 当n=2时，有`00, 01, 10`共3个；  
* 当n=3时，有`000, 001, 010, 100, 101`共5个。  

其实可以看到`f(2)=3, f(3)=5`，即`f(XXX)=f(0XX)+f(10X)`，即`f(3)=f(2)+f(1)`。

因此，放到这题里，就是需要求出小于等于某个数的所有结果。举例：  

* 当`num=100`时，即`num=1100100`，那么`1100100 = 0000000~0111111 + 1000000~1011111 + 1100000~1100011 + 1100100`，由于`1100000`已经有两个1，因此后面的都可以忽略，即`1100100 = 0000000~0111111 + 1000000~1011111`，即`res=f(6)+f(5)=34`
* 当`num=9`时，即`num=1001`，那么`1001 = 0000~0111 + 1000~1000 + 1001`，即`res=f(3)+f(0)+1=7`

#### 代码

```cpp
class Solution {
public:
    int findIntegers(int num) {
        int fib[32] = {0};
        fib[0] = 1;
        int pre = 1;
        for(int i=1;i<32;i++) {
            fib[i] = fib[i-1] + pre;
            pre = fib[i-1];
        }
        string bin = "";
        while(num) {
            bin = bin + to_string(num&1);
            num >>= 1;
        }
        int res = 0;
        int pre_bit = 0;
        for(int i=bin.size()-1;i>=0;i--) {
            if(bin[i] == '1') {
                res += fib[i];
                if(pre_bit == 1)return res;
                pre_bit = 1;
                // cout<<i<<" "<<fib[i]<<endl;
            }
            else
                pre_bit = 0;
        }
        return 1 + res;
    }
};
```

# 4. Median of Two Sorted Arrays

#### 题目

https://leetcode.com/problems/median-of-two-sorted-arrays/

#### 思路

思路参考链接：https://discuss.leetcode.com/topic/4996/share-my-o-log-min-m-n-solution-with-explanation

就是说，需要先明确中值的意义：找到某个值，使得比这个值小的数和比这个数大的数的个数相同。

* 首先确认长度最小的数组`n1`，对它进行二分查找。因为如果对长序列查找，那么对应短序列的index会出错；
* 如果当前在`n1`的index为`i`，那么考虑`len1+len2-i-j==i+j-1`，即左侧长度和右侧长度一样，那么在`n2`的index为`j=(len1+len2+1)/2-i`；
* 当`n1[i-1] <= n2[j] && n1[i] >= n2[j-1]`时，就找到了对应的`i`，此时中值为`m+n为奇数：max(n1[i-1],n2[j-1]); m+n为偶数：(max(n1[i-1],n2[j-1])+min(n1[i],n2[j]))/2`
* 当`n1[i-1] > n2[j]`时，`right=i-1`；
* 当`n1[i] < n2[j-1]`时，`left=i+1`.



#### 代码

```cpp
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int len1 = nums1.size();
        int len2 = nums2.size();
        vector<int> n1 = nums1, n2 = nums2;
        if(len1 > len2) {
            n1 = nums2;
            n2 = nums1;
            swap(len1, len2);
        }
        if(len1 == 0) {
            return (n2[len2/2] + n2[(len2-1)/2]) / 2.;
        }
        int l = 0;
        int r = len1;
        int i, j;
        // cout<<len1<<" "<<len2<<endl;
        while(l <= r) {
            i = (l + r) / 2;
            j = (len1+len2+1) / 2 - i;
            if(i > 0 && n1[i-1] > n2[j]) {
                r = i-1;
            }
            else if(i < len1 && n2[j-1] > n1[i]) {
                l = i+1;
            }
            else
                break;
        }
        int maxlef;
        if(i == 0)
            maxlef = n2[j-1];
        else if(j == 0)
            maxlef = n1[i-1];
        else
            maxlef = max(n1[i-1], n2[j-1]);
        int minright;
        if(j == len2)
            minright = n1[i];
        else if(i == len1)
            minright = n2[j];
        else
            minright = min(n1[i], n2[j]);
        if((len1 + len2) % 2 == 1)
            return maxlef;
        else
            return ((double)maxlef + (double)minright) / 2.;
    }
};
```

# 632. Smallest Range

#### 题目

https://leetcode.com/problems/smallest-range/description/

#### 思路

考虑最差的情况就是所有有序数组的最小值中的最小值作为range左侧，最大值作为range右侧。使用优先级队列按有序数组的第一个数进行排列，每次pop出队列中最小的值，并更新range。直到某个数组里的数pop完为止，此时表明当前range不包含所有的数组。

#### 代码

```cpp
typedef vector<int>::iterator vit;
class Solution {
public:
    struct comp {
        bool operator() (pair<vit, vit> a, pair<vit, vit> b) {
            return *a.first > *b.first;
        }
    };
    vector<int> smallestRange(vector<vector<int>>& nums) {
        priority_queue<pair<vit, vit>, vector<pair<vit, vit>>, comp> pq;
        int low = INT_MAX;
        int high = INT_MIN;
        for(int i=0;i<nums.size();i++) {
            low = min(low, nums[i][0]);
            high = max(high, nums[i][0]);
            pq.push(make_pair(nums[i].begin(), nums[i].end()));
        }
        vector<int> res = {low, high};
        while(1) {
            auto head = pq.top();
            pq.pop();
            ++head.first;
            if(head.first == head.second)
                break;
            pq.push(head);
            low = *pq.top().first;
            high = max(high, *head.first);
            if(high - low < res[1] - res[0])
                res = {low, high};
        }
        return res;
    }
};
```

# 768. Max Chunks To Make Sorted II

#### 题目

https://leetcode.com/problems/max-chunks-to-make-sorted-ii/description/

#### 思路

其实还是找每个值的排序下标（pos）与在原数组中的下标（arr），那么对于某个数，其包含的排序线段范围为[min(i, pos[i]), max(pos[i], i)]，因此合并重叠的线段，最终得到的不重叠的线段个数即为结果。

代码中简化了找非重叠线段。

#### 代码

```cpp
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        int n = arr.size();
        vector<int> pos(n);
        iota(pos.begin(), pos.end(), 0);
        sort(pos.begin(), pos.end(), [&arr](int a, int b){return arr[a] == arr[b] ? a < b : arr[a] < arr[b];});
        int max_n = 0;
        int res = 0;
        for(int i=0;i<n;i++) {
            max_n = max(max_n, pos[i]);
            if(max_n == i) res++;
        }
        return res;
    }
};
```
