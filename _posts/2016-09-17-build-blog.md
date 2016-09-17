--- 
layout: post 
title: 利用Github Pages搭建独立博客
date: 2016-09-17 
categories: blog 
tags: [系统搭建, 独立博客] 
description: 搭建blog
--- 

# 利用Github Pages搭建独立博客

Github Pages一开始好像是用来做github项目管理的，当然由于其可以自定义主页，即index.html，所以可以利用它来搭建single page的博客。再配合上github上的开源项目Jekyll，丰富了博客的功能。

### 搭建流程
* [创建github账号](https://github.com/)和[github pages项目](https://pages.github.com/)
* 配置本地github环境（默认已经搭建好，如果没有，请参考[windows下git环境搭建](http://blog.sina.com.cn/s/blog_a5191b5c0102v4w6.html)或者[mac下git环境搭建](http://www.cnblogs.com/heyonggang/p/3462191.html)）
* 创建/导入Jekyll模板
* 学习Jekyll的工作原理并修改模板
* 注册、解析与绑定域名

### 预备知识

* github的基本命令操作（如创建项目，拉分支以及同步线上项目）

### 1.创建github账号与github pages项目

这一节默认大家都已经创建好了github账号，如果没有，请戳上面的链接。

github pages项目可以通过创建项目（create repository）来进行，并令项目名为`username.github.io`，其中`username`就是你的github账号用户名。
创建完之后可以通过下面命令将该项目拉到本地：

```
git init # 初始化本地代码仓库
git add ./ # 添加修改的项目
git commit -m "refresh" # 将修改的内容融入到本地代码仓库
git pull origin master # 本地同步线上，如果线上没有修改，可以忽略
git push -u origin master # 将本地代码仓库同步到线上
```

### 2.创建/导入Jekyll模板
