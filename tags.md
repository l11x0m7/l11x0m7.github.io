---
layout: page
title: "Tags"
description: "物以类聚 人以群分"
header-img: "img/semantic.jpg"
---

<link rel="stylesheet" href="/css/tag-graph.css">

## 标签知识图谱

每个节点是一个标签,节点越大代表文章越多;两个标签出现在同一篇文章就会连线,线越粗代表它们越经常「同框」。

**玩法**:点击节点查看相关文章 · 拖拽节点重新布局 · 滚轮缩放 · 点空白处重置

<div id="tag-graph-wrap">
  <div id="tag-graph">
    <svg id="tag-graph-svg" role="img" aria-label="标签共现知识图谱"></svg>
    <div id="tag-graph-tip"></div>
    <div id="tag-graph-noscript">需要启用 JavaScript 才能渲染知识图谱</div>
  </div>

  <aside id="tag-graph-panel">
    <p class="tg-panel-hint">点击一个标签节点,这里会列出它的文章</p>
  </aside>
</div>

<!-- 图谱数据:由 Jekyll 在构建时输出,JS 读取后计算共现并渲染 -->
<script type="application/json" id="tag-graph-data">
{
  "tags": [
    {% for tag in site.tags %}
    { "name": {{ tag[0] | jsonify }}, "count": {{ tag[1].size }} }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ],
  "posts": [
    {% for post in site.posts %}
    { "title": {{ post.title | jsonify }}, "url": "{{ post.url | prepend: site.baseurl }}", "date": "{{ post.date | date: '%Y-%m-%d' }}", "tags": {{ post.tags | jsonify }} }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ]
}
</script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js" defer></script>
<script src="/js/tag-graph.js" defer></script>
