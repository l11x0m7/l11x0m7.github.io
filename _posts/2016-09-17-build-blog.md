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

1. 自行创建Jekyll项目，可以参考[一步步在GitHub上创建博客主页](http://www.pchou.info/ssgithubPage/2013-01-03-build-github-blog-page-01.html)。

2. 直接fork别人的Jekyll模板。我用的是[这里](https://github.com/cnfeat/blog.io)的模板。将fork到自己账号下的模板（项目里的所有内容）全都复制到我们上面创建的项目里。

然后直接登录`https://username.github.io`即可看到一个成熟的独立博客啦！

### 3.学习Jekyll的工作原理并修改模板（参考：[一步步在GitHub上创建博客主页](http://www.pchou.info/ssgithubPage/2013-01-03-build-github-blog-page-01.html)）

jekyll是一个基于ruby开发的，专用于构建静态网站的程序。它能够将一些动态的组件：模板、liquid代码等构建成静态的页面集合，Github-Page全面引入jekyll作为其构建引擎，这也是学习jekyll的主要动力。同时，除了jekyll引擎本身，它还提供一整套功能，比如web server。我们用jekyll –server启动本地调试就是此项功能。读者可能已经发现，在启动server后，之前我们的项目目录下会多出一个_site目录。jekyll默认将转化的静态页面保存在_site目录下，并以某种方式组织。使用jekyll构建博客是十分适合的，因为其内建的对象就是专门为blog而生的，在后面的逐步介绍中读者会体会到这一点。但是需要强调的是，jekyll并不是博客软件，跟workpress之类的完全两码事，它仅仅是个一次性的模板解析引擎，它不能像动态服务端脚本那样处理请求。

更多关于jekyll请看[这里](http://jekyllbootstrap.com/lessons/jekyll-introduction.html)。是jekyllbootstrap的官网，里面有比较详细的原理介绍。

了解原理后，我们看看Jekyll的工程结构（以我使用的模板为例）：

```
.
├── 404.html # 访问时产生404错误时返回的页面
├── CNAME # 里面保存着域名，如果买过域名的话，可以通过这个来绑定
├── Gruntfile.js
├── Interview.html
├── LICENSE
├── README.md
├── _config.yml # 配置文件
├── _includes # 类似c++的include头文件
│   ├── footer.html # 设置你的博客（包括所有）的底部内容
│   ├── head.html # 设置你的博客的顶部内容
│   └── nav.html # 设置你的博客的标签内容（上方右边）
├── _layouts # 博客主要的显示格式
│   ├── default.html # 默认格式，类似于类里的基类
│   ├── page.html # 每个标签（如home、tags）下的格式
│   └── post.html # 每篇博文的格式
├── _posts # 存放你提交的博文，注意命名格式
│   ├── 2016-09-15-mid-moon-fest.md
│   ├── 2016-09-16-happy-day.md
│   ├── 2016-09-17-build-blog.md
│   └── 2016-09-17-reservoir-sampling.md
├── about.md # about页面的数据
├── archive.md # archive页面的数据
├── css # 这个不用管，基本是一些额外功能的样式表，如代码高亮、latex公式等。
│   ├── backtop.css
│   ├── bootstrap.css
│   ├── bootstrap.min.css
│   ├── clean-blog.css
│   ├── clean-blog.min.css
│   └── syntax.css
├── feed.xml
├── fonts # 字体，不用管
│   ├── glyphicons-halflings-regular.eot
│   ├── glyphicons-halflings-regular.svg
│   ├── glyphicons-halflings-regular.ttf
│   ├── glyphicons-halflings-regular.woff
│   └── glyphicons-halflings-regular.woff2
├── img # 存放图片资源文件
│   ├── Back-Top_Arrow.png
│   ├── black.jpg
│   ├── blue.jpg
│   ├── dolphin.gif
│   ├── facebook.jpg
│   ├── favicon.ico
│   ├── favicon.png
│   ├── green.jpg
│   ├── orange.jpg
│   ├── red.jpg
│   ├── semantic.jpg
│   ├── skyhigh.ico
│   ├── skyhigh1107.jpg
│   ├── twitter.jpg
│   └── zhihu.jpg
├── index.html # 博客的主页
├── js
│   ├── backtop.js
│   ├── bootstrap.js
│   ├── bootstrap.min.js
│   ├── clean-blog.js
│   ├── clean-blog.min.js
│   ├── jquery.js
│   └── jquery.min.js
├── less
│   ├── clean-blog.less
│   ├── mixins.less
│   └── variables.less
├── mkblog.sh # 我自己写的创建博客的脚本，注意创建的博文文件命名方式以及开头的信息说明
├── package.json
├── tags.md # tags页面的数据
├── upload.sh # 我自己写的上传博文的脚本
└── zone.md # zone页面的数据
```

上面没注释的可以不用管，不妨碍使用。主要的几个文件的说明：

* _config.yml：保存配置，该配置将影响jekyll构造网站的各种行为。
* \_includes：该目录下的文件可以用来作为公共的内容被其他文章引用，就跟C语言include头文件的机制完全一样，jekyll在解析时会对标记<span>{</span><span>%</span> <span>include</span> <span>%</span><span>}</span>扩展成对应的在\_includes文件夹中的文件。
* _layouts：该目录下的文件作为主要的模板文件。
* _posts：文章或网页应当放在这个目录中，但需要注意的是，文章的文件名必须是YYYY-MM-DD-title，里面可以放html文件，也可以放markdown文件。
* _site：这是jekyll默认的转化结果存放的目录，我的模板里没用到。
* assets：这个目录没有强制的要求，主要目的是存放你的资源文件，图片、样式表、脚本等，我的模板里没用到。

### 4.注册、解析与绑定域名（可选）

关于域名的科普，请戳[这里](http://www.pchou.info/ssgithubPage/2013-01-05-build-github-blog-page-03.html)。

可以到[dnspod](https://www.dnspod.cn/)或者[godaddy](https://sg.godaddy.com/zh)上购买域名，不过好像dnspod购买类似.com或.cn的域名还需要实名认证，需要审核材料等，比较浪费时间，可以直接在godaddy上购买域名，不用繁琐的步骤即可使用。

购买域名后，需要对域名进行解析，解析的目的是把你的域名放到dns服务器上，这样你在浏览器中输入网址的时候才能够响应。但是如何才能够和github上的地址关联起来呢？可以在域名解析页面配置如下（图片截取自我在dnspod购买的域名解析网页）：

<img src="http://bloglxm.oss-cn-beijing.aliyuncs.com/%E5%9F%9F%E5%90%8D%E8%A7%A3%E6%9E%90.png">

如果上面添加的A记录（域名绑定IP）不能生效，则可以添加一条CNAME记录（域名绑定域名）。格式为：

主机记录：@	记录类型：CNAME	记录值：username.github.io

之后就是修改上面说你的github项目里的CNAME（这里的CNAME文件是要和域名解析时填写的记录对应）了，把你的域名写入CNAME文件即可。比如我的CNAME文件里写的是：[skyhigh233.com](skyhigh233.com)。


### 5.总结
虽然这篇博客写的比较粗浅，但是把主要的步骤都说明了一下，并贴出了具体的链接。总的来说，使用github来搭建博客，功能不会像WordPress那样强大，或者是自己购买服务器搭建web服务那样灵活。但是正是因为有很多内容与机制都被Jekyll封装起来了，我们也没必要花费太大的精力在里面，毕竟我们主要的目的是用它来存放我们平时写的博客嘛，哈哈。因此省去了后端的操作，以及直接使用前端的模板，这样就把整个过程的工作量大大降低了，我们要做的就是了解流程和原理，知道如何去修改和配置文件以达到我们想要的目的。关于功能的扩建，比如站内搜索、流量统计等，如果需要的话，也是可以加进去的，这里就不赘述了。

希望这篇文章对大家有用。谢谢！！