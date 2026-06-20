---
layout: page
title: "Archive"
description: "不积跬步 无以至千里"
header-img: "img/orange.jpg"
---

<link rel="stylesheet" href="/css/archive-timeline.css">

<div class="cyber-timeline">
{% for post in site.posts %}
  {% capture y %}{{ post.date | date:"%Y" }}{% endcapture %}
  {% if year != y %}
    {% assign year = y %}
  <div class="tl-year"><span class="tl-year-label">{{ y }}</span></div>
  {% endif %}
  <div class="tl-item">
    <span class="tl-dot"></span>
    <div class="tl-card">
      <time class="tl-date" datetime="{{ post.date | date:"%Y-%m-%d" }}">{{ post.date | date:"%m-%d" }}</time>
      <a class="tl-title" href="{{ post.url | prepend: site.baseurl }}" title="{{ post.title }}">{{ post.title }}</a>
    </div>
  </div>
{% endfor %}
</div>

<script src="/js/archive-timeline.js" defer></script>
