--- 
layout: post 
title: LDA主题模型之算法实现
date: 2016-10-20 
categories: blog 
tags: [NLP, LDA] 
description: LDA模型
--- 

# LDA主题模型之算法实现

## 写在前面

希望看这篇文章的读者对LDA主题模型的原理有所了解，不然可能会比较吃力。具体的原理解析可以参考：

* [LDA数学八卦](http://www.52nlp.cn/lda-math-%E6%B1%87%E6%80%BB-lda%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6)
* [LDA漫游指南](http://yuedu.baidu.com/ebook/d0b441a8ccbff121dd36839a)

如果文章有错误的地方，希望能得到您的指正，谢谢。

## 模型比较

LDA模型（Latent Dirichlet Model）用于寻找多个文档内存在的多个主题的模型。和LSA不同的是，LSA需要使用SVD找到的隐形变量是未知属性的。也就是说，LSA后文档或词的每一维所代表的含义不明确。  
而对于PLSA模型，它主要是考虑到了文档和词汇之间的关系是通过主题来联系的。即一篇文档的生成过程为：

* 按概率选择多个主题中的一个主题（多项分布）
* 在该主题下，按概率选择该主题中的某个词（另一个多项分布）

那么，生成的文档概率正比于两个多项分布的乘积。可以通过EM算法求解局部最优解。而LDA则是在PLSA的思路上进行扩充，给每个多项分布设定一个参数的先验分布。这样我们可以求得参数的后验分布，进一步可以求得w和z的联合分布。

## 原理概述

具体原理之后有时间另开一章，现在说明大致过程。

LDA的算法是基于Gibbs Sampling算法的，Gibbs Sampling算法又是基于MCMC的。LDA的目的是获得满足w和z的联合分布的样本点（词的主题）。而Gibbs Sampling就是通过迭代某个维度的条件概率（每个维度对应某个文档的某个位置的词）获得平稳状态，而这平稳状态的分布即这条件概率对应的联合概率。所以我们可以通过这种方法，得到稳定分布的样本点。  
我们需要迭代的条件分布为：

$$p(z_i=k|\overrightarrow z_{\neg i}, \overrightarrow w)\propto\frac{n^{(k)}_{m,\neg i}+\alpha_k}{\sum^K_{k=1}(n_{m,\neg i}^{(k)}+\alpha_k)}*\frac{n^{(t)}_{k,\neg i}+\beta_t}{\sum^V_{t=1}(n_{k,\neg i}^{(t)}+\beta_t)}=\frac{n^{(k)}_{m,\neg i}+\alpha_k}{\sum^K_{k=1}n_{m,\neg i}^{(k)}+\sum^K_{k=1}\alpha_k}*\frac{n^{(t)}_{k,\neg i}+\beta_t}{\sum^V_{t=1}n_{k,\neg i}^{(t)}+\sum^V_{t=1}\beta_t}$$

其中：  
$n^{(k)}_{m,\neg i}$表示文档m中主题为k的词数（不包含当前词i）  

$\sum^K_{k=1}n_{m,\neg i}^{(t)}$表示文档m的总词数（不包含当前词i）  

$n^{(t)}_{k,\neg i}$表示主题k下词汇t的词频（不包含当前词i）  

$\sum^V_{t=1}n_{k,\neg i}^{(t)}$表示主题k的总词数（不包含当前词i）

在给出代码前，需要注意：

* 一篇文档对应多个词，多个主题
* 一个词汇对应多个主题（注意词和词汇的区别）
* 某篇文档下的某个词对应一个主题
* 一个主题对应多个词汇

## 代码

下面给出完整的代码，注释已给出。代码地址为：  
[l11x0m7.github:LDA](https://github.com/l11x0m7/LDA)

```python
# -*- encoding:utf-8 -*-
import os
import time
import sys
import re
import random
import numpy as np
import copy
reload(sys)
sys.setdefaultencoding('utf-8')

# 新闻爬虫
# 爬取新闻的内容有:社会新闻20篇,军事新闻20篇,国际新闻20篇
class NewsScrapy():
    def __init__(self):
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_3) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.54 Safari/536.5'
        self.headers = {'User-Agent': user_agent}


    def download(self, url, num, savepath, l_url):
        import urllib
        import urllib2
        from lxml import etree

        if not os.path.exists(savepath[:savepath.rfind('/')]):
                os.mkdir(savepath[:savepath.rfind('/')])

        if not os.path.exists(l_url) or os.stat(l_url).st_size < 10:
            fw = open(l_url, 'w')
            url_list = set()
            try:
                req = urllib2.Request(url, headers=self.headers)
                res = urllib2.urlopen(req)
                content = res.read().decode('gbk', errors='ignore')
            except Exception as e:
                print '[Error]', e.message
            root = etree.HTML(content)
            href = root.xpath('//@href')
            for h in href:
                # if len(url_list) >= num:
                #     break
                if re.match(r'http://news.qq.com/.*', h):
                    url_list.add(h)
            for url in url_list:
                fw.write(url+'\n')
            fw.close()
            print '\n'.join(url_list)
            print len(url_list)
            url_list = list(url_list)
        else:
            fr = open(l_url)
            url_list = list()
            for line in fr:
                url_list.append(line.strip())
            print 'Load successful!'
            fr.close()

        time.sleep(1)

        with open(savepath, 'w') as fw:
            try:
                count = 0
                for url in url_list:
                    req = urllib2.Request(url, headers=self.headers)
                    res = urllib2.urlopen(req)
                    content = res.read().decode('gbk', errors='ignore')
                    # r = etree.HTML(content)
                    title = re.findall(r'<h1>(.*?)</h1>', content, re.S)
                    if len(title)<1:
                        continue
                    title = title[0].strip()
                    body = re.findall(r'<div id="Cnt-Main-Article-QQ" .*?<P.*?>(.*?)</div>',
                            content, re.S)
                    if len(body)<1:
                        continue
                    body = body[0].replace('</P>', "").replace('<P>', "").replace(' ', "").strip()
                    body = re.sub(r'<.*?>', "", body)
                    print title
                    print body
                    fw.write(title+'\t'+body+'\n')
                    count += 1
                    if count>=num:
                        break
                    time.sleep(5)
            except Exception as e:
                print '[Error2]', e.message if e.message!="" else url


# LDA模型
class LDAModel():
    def __init__(self, K, copora, alpha=None, beta=None, iteration=None):
        # K个主题
        self.K = K
        # alpha工程取值一般为50/K
        self.alpha = alpha if alpha else 50.0/self.K
        # beta工程取值一般为0.01
        self.beta = beta if beta else 0.01
        # 迭代次数一般取值为1000
        self.iteration = iteration if iteration else 1000
        self.nw = object    # K*V   每个主题下的每个词的出现频次
        self.nd = object    # D*K   每个文档下每个主题的词数
        self.nwsum = object # K     每个主题的总词数
        self.ndsum = object # D     每个文档的总词数
        self.theta = object # doc->topic    D*K
        self.phi = object   # topic->word   K*V
        self.z = object     # (m,w)对应每篇文档的每个词的具体主题
        # 3个语料整合到一起
        self.corpora = list()
        for theme in copora:
            with open(theme) as fr:
                for line in fr:
                    body = line.strip().split('\t')[1].decode()
                    self.corpora.append(body)

        # 文档数
        self.D = len(self.corpora)
        cut_docs = self.cut(self.corpora)
        # 分词并且id化的文档
        self.word2id, self.id2word, self.id_cut_docs = self.createDict(cut_docs)
        self.V = len(self.id2word)
        # 初始化参数
        self.initial(self.id_cut_docs)
        # gibbs采样,进行文本训练
        self.gibbsSamppling()

    # 文档分词,去无用词
    # 可以考虑去除文本低频词
    def cut(self, docs):
        from jieba import cut
        stop_words = self.loadStopWords()
        cut_docs = list()
        for doc in docs:
            cut_doc = cut(doc)
            new_doc = list()
            for word in cut_doc:
                if len(word.decode())>=2 and word not in stop_words and not word.isdigit():
                    new_doc.append(word)
            cut_docs.append(new_doc)
        return cut_docs

    # 创建word2id,id2word和document字典
    def createDict(self, cut_docs):
        word2id = dict()
        for i, doc in enumerate(cut_docs):
            for j, word in enumerate(doc):
                if not word2id.has_key(word):
                    word2id[word] = len(word2id)
                cut_docs[i][j] = word2id[word]
        return word2id, dict(zip(word2id.values(), word2id.keys())), cut_docs

    # 初始化参数
    def initial(self, id_cut_docs):
        self.nd = np.array(np.zeros([self.D, self.K]), dtype=np.int32)
        self.nw = np.array(np.zeros([self.K, self.V]), dtype=np.int32)
        self.ndsum = np.array(np.zeros([self.D]), dtype=np.int32)
        self.nwsum = np.array(np.zeros([self.K]), dtype=np.int32)
        self.z = np.array(np.zeros([self.D, self.V]), dtype=np.int32)
        self.theta = np.ndarray([self.D, self.K])
        self.phi = np.ndarray([self.K, self.V])
        # 给每篇文档的每个词随机分配主题
        for i, doc in enumerate(id_cut_docs):
            for j, word_id in enumerate(doc):
                theme = random.randint(0, self.K-1)
                self.z[i,j] = theme
                self.nd[i,theme] += 1
                self.nw[theme,word_id] += 1
                self.ndsum[i] += 1
                self.nwsum[theme] += 1

    # gibbs采样
    def gibbsSamppling(self):
        for iter in range(self.iteration):
            for i, doc in enumerate(self.id_cut_docs):
                for j, word_id in enumerate(doc):
                    theme = self.z[i,j]
                    nd = self.nd[i,theme] - 1
                    nw = self.nw[theme,word_id] - 1
                    ndsum = self.ndsum[i] - 1
                    nwsum = self.nwsum[theme] - 1
                    # 重新给词选择新的主题
                    new_theme = self.reSamppling(nd, nw, ndsum, nwsum)

                    self.nd[i,theme] -= 1
                    self.nw[theme,word_id] -= 1
                    self.nwsum[theme] -= 1

                    self.nd[i,new_theme] += 1
                    self.nw[new_theme,word_id] += 1
                    self.nwsum[new_theme] += 1
                    self.z[i,j] = new_theme
            sys.stdout.write('\rIteration:{0} done!'.format(iter+1))
            sys.stdout.flush()
        # 更新theta和phi
        self.updatePara()

    # 更新theta和phi
    def updatePara(self):
        for d in range(self.D):
            for k in range(self.K):
                self.theta[d,k] = float(self.nd[d,k] + self.alpha)/\
                                (self.ndsum[d] + self.alpha*self.K)

        for k in range(self.K):
            for v in range(self.V):
                self.phi[k,v] = float(self.nw[k,v] + self.beta)/\
                                (self.nwsum[k] + self.beta*self.K)

    # 重新选择主题
    def reSamppling(self, nd, nw, ndsum, nwsum):
        pk = np.ndarray([self.K])
        for i in range(self.K):
            # gibbs采样公式
            pk[i] = float(nd + self.alpha)*(nw +self.beta)/\
                    ((ndsum + self.alpha*self.K)*(nwsum + self.beta*self.V))
            if i > 0:
                pk[i] += pk[i-1]

        # 轮盘方式随机选择主题
        u = random.random()*pk[self.K-1]
        for k in range(len(pk)):
            if pk[k]>=u:
                return k

    # new_doc应该为list
    # 预测新的文档的主题
    def predict(self, new_doc, isupdate=False):
        new_cut_doc = self.cut(new_doc)[0]
        new_cut_id_doc = list()
        for word in new_cut_doc:
            if word in self.word2id:
                new_cut_id_doc.append(self.word2id[word])

        # 原为D*K,现在为1*K
        new_nd = np.zeros([self.K], dtype=np.int32)
        # 原为K*V,现在依然为K*V
        new_nw = copy.deepcopy(self.nw)
        # 原为1*D,现为1
        new_ndsum = 0
        # 原为1*K,现为1*K
        new_nwsum = copy.deepcopy(self.nwsum)
        # 当前文档的各个词的主题,1*V
        new_z = np.zeros([self.V]);

        # 类似于initial函数
        for j, word_id in enumerate(new_cut_id_doc):
            theme = random.randint(0, self.K-1)
            new_nd[theme] += 1
            new_nw[theme, word_id] += 1
            new_ndsum += 1
            new_nwsum[theme] += 1
            new_z[j] = theme

        # 类似于gibbsSampling函数
        for iter in range(self.iteration):
            for j, word_id in enumerate(new_cut_id_doc):
                theme = new_z[j]
                cur_nd = new_nd[theme] -1
                cur_nw = new_nw[theme,word_id] - 1
                cur_ndsum = new_ndsum - 1
                cur_nwsum = new_nwsum[theme] - 1

                new_theme = self.reSamppling(cur_nd, cur_nw, cur_ndsum, cur_nwsum)

                new_z[j] = new_theme
                new_nd[theme] -= 1
                new_nw[theme, word_id] -= 1
                new_nwsum[theme] -= 1

                new_nd[new_theme] += 1
                new_nw[new_theme,word_id] += 1
                new_nwsum[new_theme] += 1

            sys.stdout.write('\rIteration:{0} done!'.format(iter+1))
            sys.stdout.flush()

        # 更新theta参数,即再加1行
        new_theta = np.ndarray([self.K])
        for k in range(self.K):
            new_theta[k] = float(new_nd[k] + self.alpha)/\
                            (new_ndsum + self.alpha*self.K)

        # 若更新phi参数,即更新phi矩阵中所有的参数
        # 则需要同时更新theta,z,nw,nd,nwsum,ndsum
        # 也可以不更新,即参数和原来没预测之前一样
        if isupdate:
            for k in range(self.K):
                for v in range(self.V):
                    self.phi[k,v] = float(new_nw[k,v] + self.beta)/\
                                    (new_nwsum[k] + self.beta*self.K)

        # 返回该文档的各个主题出现概率,以及每个词对应的主题
        return new_theta, new_z


    # 返回各个主题的top词汇
    def getTopWords(self, top_num=20):
        with open('./data/result/topwords', 'w') as fw:
            for k in range(self.K):
                top_words = np.argsort(-self.phi[k,:])[:top_num]
                top_words = [self.id2word[word] for word in top_words]
                top_words = '\t'.join(top_words)
                res = 'topic{0}\t{1}'.format(k, top_words)
                fw.write(res+'\n')
                print res



    # 返回文档前几个topic中的前几个词
    def getTopTopics(self, top_topic=5, top_word=5):
        with open('./data/result/toptopics', 'w') as fw:
            for d in range(self.D):
                top_topics = np.argsort(-self.theta[d,:])[:top_topic]
                print 'document{0}:'.format(d)
                for topic in top_topics:
                    top_words_id = np.argsort(-self.phi[topic,:])[:top_word]
                    top_words = [self.id2word[word] for word in top_words_id]
                    top_words = '\t'.join(top_words)
                    res = 'topic{0}\t{1}'.format(topic, top_words)
                    fw.write(res+'\n')
                    print res
                fw.write('\n')


    # 载入停用词
    def loadStopWords(self):
        # 停用词：融合网络停用词、哈工大停用词、川大停用词
        root_path = '..'
        stop_words = set()
        with open(root_path + u'/Dict/StopWords/file/中文停用词库.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/哈工大停用词表.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/四川大学机器智能实验室停用词库.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/百度停用词列表.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/stopwords_net.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/stopwords_net2.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        stop_words.add('')
        stop_words.add(' ')
        stop_words.add(u'\u3000')
        stop_words.add(u'日')
        stop_words.add(u'月')
        stop_words.add(u'时')
        stop_words.add(u'分')
        stop_words.add(u'秒')
        stop_words.add(u'报道')
        stop_words.add(u'新闻')
        return stop_words




if __name__ == '__main__':
    crawer = NewsScrapy()
    # crawer.download('http://news.qq.com/society_index.shtml', 20, './data/society', './data/society_urls')
    # crawer.download('http://mil.qq.com/mil_index.htm', 20, './data/military', './data/military_urls')
    # crawer.download('http://news.qq.com/world_index.shtml', 20, './data/world', './data/world_urls')

    corpus = ['./data/society', './data/military', './data/world']
    time1 = time.time()
    ldamodel = LDAModel(20, corpus, iteration=300)
    time2 = time.time()
    print 'Training time:{0}'.format(time2-time1)
    # 各个主题的top20词汇
    print '每个主题的top words'
    ldamodel.getTopWords(20)
    # 各个文档的top5话题,每个话题的top5词汇
    print '每篇文档的top topics中的top words'
    ldamodel.getTopTopics()

    time1 = time.time()
    # 预测一篇文档的top5主题,返回每个主题的top5词汇
    pred_doc_id = 0
    pred_doc = [ldamodel.corpora[pred_doc_id]]
    new_theta, new_z = ldamodel.predict(pred_doc)
    time2 = time.time()
    print 'Predict time per document:{0}'.format(time2-time1)

    print '预测文档{0}:'.format(pred_doc_id)
    top_topics = np.argsort(-new_theta)[:5]
    for topic in top_topics:
        top_words_id = np.argsort(-ldamodel.phi[topic,:])[:5]
        top_words = [ldamodel.id2word[word] for word in top_words_id]
        top_words = '\t'.join(top_words)
        print 'topic{0}\t{1}'.format(topic, top_words)
```