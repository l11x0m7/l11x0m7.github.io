--- 
layout: post 
title: 树结构算法总结(1) 二叉树的遍历
date: 2016-10-01 
categories: blog 
tags: [tree, lintcode] 
description: 总结树结构的算法
--- 

# 树结构算法总结(1) 二叉树的遍历

## 非递归的遍历

### 前序遍历

#### 思路
使用栈来模拟，比较简单，直接上代码。

#### 代码

```cpp
/**
 * Definition of TreeNode:
 * class TreeNode {
 * public:
 *     int val;
 *     TreeNode *left, *right;
 *     TreeNode(int val) {
 *         this->val = val;
 *         this->left = this->right = NULL;
 *     }
 * }
 */

class Solution {
public:
    /**
     * @param root: The root of binary tree.
     * @return: Preorder in vector which contains node values.
     */
    vector<int> preorderTraversal(TreeNode *root) {
        // write your code here
        stack<TreeNode*> q;
        vector<int> res;
        TreeNode* cur = root;
        while(cur){
            res.push_back(cur->val);
            q.push(cur);
            cur = cur->left;
        }
        while(!q.empty()){
            TreeNode* cur = q.top();
            q.pop();
            if(cur->right){
                cur = cur->right;
                while(cur){
                    res.push_back(cur->val);
                    q.push(cur);
                    cur = cur->left;
                }
            }
            
        }
        return res;
    }
};
```

### 中序遍历

#### 思路

同样用栈模拟，对比前序遍历稍作修改即可。

#### 代码

```cpp
/**
 * Definition of TreeNode:
 * class TreeNode {
 * public:
 *     int val;
 *     TreeNode *left, *right;
 *     TreeNode(int val) {
 *         this->val = val;
 *         this->left = this->right = NULL;
 *     }
 * }
 */
class Solution {
    /**
     * @param root: The root of binary tree.
     * @return: Inorder in vector which contains node values.
     */
public:
    vector<int> inorderTraversal(TreeNode *root) {
        // write your code here
        stack<TreeNode*> s;
        vector<int> res;
        TreeNode* cur = root;
        while(cur){
            s.push(cur);
            cur = cur->left;
        }
        while(!s.empty()){
            TreeNode* cur = s.top();
            s.pop();
            res.push_back(cur->val);
            cur = cur->right;
            while(cur){
                s.push(cur);
                cur = cur->left;
            }
        }
        return res;
    }
};
```

#### 扩展——二叉查找树迭代器

二叉查找树是用于查找的平衡二叉排序树。可以使用中序遍历的思路来按顺序（数值的大小）访问节点。

代码如下：

```cpp
/**
 * Definition of TreeNode:
 * class TreeNode {
 * public:
 *     int val;
 *     TreeNode *left, *right;
 *     TreeNode(int val) {
 *         this->val = val;
 *         this->left = this->right = NULL;
 *     }
 * }
 * Example of iterate a tree:
 * BSTIterator iterator = BSTIterator(root);
 * while (iterator.hasNext()) {
 *    TreeNode * node = iterator.next();
 *    do something for node
 */
class BSTIterator {
public:
    stack<TreeNode*> s;
    //@param root: The root of binary tree.
    BSTIterator(TreeNode *root) {
        // write your code here
        while(root){
            s.push(root);
            root = root->left;
        }
    }

    //@return: True if there has next node, or false
    bool hasNext() {
        // write your code here
        return !s.empty();
    }
    
    //@return: return next node
    TreeNode* next() {
        // write your code here
        TreeNode* cur = s.top();
        s.pop();
        TreeNode* tmp = cur;
        tmp = tmp->right;
        while(tmp){
            s.push(tmp);
            tmp = tmp->left;
        }
        return cur;
    }
};
```

### 后序遍历

#### 思路

同样使用栈来模拟，不过要考虑的顺序是左、右、中。首先存储树的左边节点，之后判定最后一个左节点是否有右节点，如果有的话先对该右节点做上面同样的操作；如果没有右节点或者右节点被访问过，则访问当前节点，并将该节点从栈中删除。

#### 代码

```cpp
/**
 * Definition of TreeNode:
 * class TreeNode {
 * public:
 *     int val;
 *     TreeNode *left, *right;
 *     TreeNode(int val) {
 *         this->val = val;
 *         this->left = this->right = NULL;
 *     }
 * }
 */
class Solution {
    /**
     * @param root: The root of binary tree.
     * @return: Postorder in vector which contains node values.
     */
public:
    vector<int> postorderTraversal(TreeNode *root) {
        // write your code here
        vector<int> res;
        stack<TreeNode*> s;
        while(root){
            s.push(root);
            root = root->left;
        }
        TreeNode* pre = NULL;
        while(!s.empty()){
            TreeNode* cur = s.top();
            if(!cur->right||cur->right==pre){
                res.push_back(cur->val);
                pre = cur;
                s.pop();
            }
            else{
                cur = cur->right;
                while(cur){
                    s.push(cur);
                    cur = cur->left;
                }
            }
        }
        return res;
    }
};
```

### 层序遍历

#### 思路 

使用队列来模拟。如果要区别每层的元素有哪些，则需要使用一个变量来存储每层的个数，比如一开始第一层只有一个根节点，故size=1。访问过第一层后，size=0，之后更新第二层size=queue.size()，即之前通过第一层访问后产生的所有左右节点的个数，即为第二层。同理，以此类推。

#### 代码

```cpp
/**
 * Definition of TreeNode:
 * class TreeNode {
 * public:
 *     int val;
 *     TreeNode *left, *right;
 *     TreeNode(int val) {
 *         this->val = val;
 *         this->left = this->right = NULL;
 *     }
 * }
 */
 
 
class Solution {
    /**
     * @param root: The root of binary tree.
     * @return: Level order a list of lists of integer
     */
public:
    vector<vector<int>> levelOrder(TreeNode *root) {
        // write your code here
        vector<vector<int>> res;
        queue<TreeNode*> q;
        // if(root==NULL)
        //     return res;
        q.push(root);
        TreeNode* cur;
        int size = 0;
        vector<int> tmp;
        while(!q.empty()){
            cur = q.front();
            if(size==0)
                size = q.size();
            q.pop();
            if(cur!=NULL){
                q.push(cur->left);
                q.push(cur->right);
                tmp.push_back(cur->val);
            }
            size--;
            if(!q.empty()&&size==0){
                res.push_back(tmp);
                tmp.clear();
            }
        }
        return res;
    }
};
```

