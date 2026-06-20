/* ============================================================
   Blog Search — 主页搜索(子串模糊 + 同义词扩展 + 命中高亮)
   读取 /search.json,用同义词表扩展 query,Fuse.js 模糊匹配
   标题/标签/描述/正文,命中处用 <mark> 标红加粗。
   不依赖 jQuery;Fuse 从 CDN 全局加载,缺失时降级提示。
   ============================================================ */
(function () {
  'use strict';

  var input = document.getElementById('blog-search-input');
  var statusEl = document.getElementById('blog-search-status');
  var resultsEl = document.getElementById('blog-search-results');
  if (!input || !resultsEl) return;

  // —— 同义词表:同组词互为等价,query 命中任一则整组加入搜索 ——
  // 按博客实际标签/术语维护;双向生效。
  var SYNONYMS = [
    ['深度学习', 'DeepLearning', 'Deep Learning', 'DL'],
    ['机器学习', 'ML', 'Machine Learning'],
    ['自然语言处理', 'NLP'],
    ['强化学习', 'RL', 'Reinforcement Learning'],
    ['神经网络', 'Neural Network'],
    ['算法', '数据结构'],
    ['排序', '快排', '快速排序'],
    ['总结', '小结', '复盘'],
    ['日记', '杂记'],
    ['影评', '电影']
  ];

  // INDEX_URL 可被全局 window.BLOG_SEARCH_INDEX_URL 覆盖(预览/测试用),默认 /search.json
  var INDEX_URL = (window.BLOG_SEARCH_INDEX_URL || '/search.json');
  var MAX_RESULTS = 15;
  var DEBOUNCE_MS = 300;

  var index = null;       // 缓存的搜索索引(原始 post 数组)
  var fuse = null;        // Fuse 实例
  var loaded = false;
  var loadFailed = false;

  // —— 加载索引 ——
  function loadIndex(cb) {
    if (loaded) { cb(); return; }
    if (loadFailed) { showStatus('搜索索引加载失败,请刷新重试', true); return; }
    fetch(INDEX_URL, { cache: 'force-cache' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) {
        if (typeof Fuse === 'undefined') {
          loadFailed = true;
          showStatus('搜索库(Fuse.js)未加载,搜索暂不可用', true);
          return;
        }
        index = data || [];
        fuse = new Fuse(index, {
          keys: [
            { name: 'title', weight: 0.4 },
            { name: 'tags', weight: 0.3 },
            { name: 'description', weight: 0.2 },
            { name: 'content', weight: 0.1 }
          ],
          includeMatches: true,
          includeScore: true,
          threshold: 0.4,          // 模糊度,中文偏高一点容错
          ignoreLocation: true,    // 长正文任意位置命中都算
          minMatchCharLength: 1,   // 支持单字中文
          findAllMatches: true
        });
        loaded = true;
        cb();
      })
      .catch(function (err) {
        loadFailed = true;
        console.error('blog-search: 索引加载失败', err);
        showStatus('搜索索引加载失败,请刷新重试', true);
      });
  }

  // —— 同义词扩展:返回所有应搜索的词(原词 + 同组词)——
  function expandQuery(q) {
    var words = [q];
    SYNONYMS.forEach(function (group) {
      group.forEach(function (term) {
        if (q.indexOf(term) !== -1) {
          group.forEach(function (t) { if (words.indexOf(t) === -1) words.push(t); });
        }
      });
    });
    return words;
  }

  // —— 用多个词搜 Fuse,合并去重,按最佳 score 排序 ——
  function searchAll(query) {
    var words = expandQuery(query.trim());
    var byUrl = {};
    words.forEach(function (w) {
      if (!w) return;
      fuse.search(w).forEach(function (res) {
        var url = res.item.url;
        if (!byUrl[url] || res.score < byUrl[url].score) {
          byUrl[url] = res; // 保留该文章最高分(最小 score)的匹配结果
        }
      });
    });
    var arr = Object.keys(byUrl).map(function (k) { return byUrl[k]; });
    arr.sort(function (a, b) { return a.score - b.score; });
    return arr.slice(0, MAX_RESULTS);
  }

  // —— 高亮:根据 Fuse 的 matches.indices 在文本里包 <mark> ——
  function highlight(text, indices) {
    if (!text) return '';
    if (!indices || !indices.length) return escapeHtml(text);
    // 合并重叠/相邻区间
    var ranges = indices.slice().sort(function (a, b) { return a[0] - b[0]; });
    var merged = [ranges[0]];
    for (var i = 1; i < ranges.length; i++) {
      var last = merged[merged.length - 1];
      if (ranges[i][0] <= last[1] + 1) {
        last[1] = Math.max(last[1], ranges[i][1]);
      } else {
        merged.push(ranges[i]);
      }
    }
    var out = '';
    var prev = 0;
    merged.forEach(function (r) {
      out += escapeHtml(text.slice(prev, r[0]));
      out += '<mark>' + escapeHtml(text.slice(r[0], r[1] + 1)) + '</mark>';
      prev = r[1] + 1;
    });
    out += escapeHtml(text.slice(prev));
    return out;
  }

  // —— 取摘要:命中 content 片段上下文 ——
  function excerpt(content, match) {
    if (!content) return '';
    var center = 0;
    if (match && match.indices && match.indices.length) {
      center = match.indices[0][0];
    }
    var radius = 70;
    var start = Math.max(0, center - radius);
    var end = Math.min(content.length, center + radius);
    var slice = content.slice(start, end);
    var prefix = start > 0 ? '…' : '';
    var suffix = end < content.length ? '…' : '';
    var indices = match && match.indices ? match.indices.map(function (r) {
      return [r[0] - start, r[1] - start];
    }) : null;
    return prefix + highlight(slice, indices) + suffix;
  }

  // —— 渲染结果 ——
  function render(results, query) {
    resultsEl.innerHTML = '';
    if (!results.length) {
      showStatus('没有找到与「' + escapeHtml(query) + '」相关的文章', false);
      return;
    }
    showStatus('找到 ' + results.length + ' 篇相关文章', false);
    var frag = document.createDocumentFragment();
    results.forEach(function (res) {
      var post = res.item;
      var matches = res.matches || [];
      var titleMatch = findMatch(matches, 'title');
      var tagsMatch = findMatch(matches, 'tags');
      var contentMatch = findMatch(matches, 'content');

      var div = document.createElement('div');
      div.className = 'blog-search-result';

      var head = document.createElement('div');
      head.className = 'bs-result-head';
      var a = document.createElement('a');
      a.className = 'bs-result-title';
      a.href = post.url;
      a.innerHTML = titleMatch
        ? highlight(post.title, titleMatch.indices)
        : escapeHtml(post.title || '');
      head.appendChild(a);
      div.appendChild(head);

      var meta = document.createElement('div');
      meta.className = 'bs-result-meta';
      meta.innerHTML = escapeHtml(post.date || '');
      var tagsHtml = renderTags(post.tags, tagsMatch);
      if (tagsHtml) meta.innerHTML += ' <span class="bs-result-tags">' + tagsHtml + '</span>';
      div.appendChild(meta);

      var ex = document.createElement('div');
      ex.className = 'bs-result-excerpt';
      ex.innerHTML = excerpt(post.content, contentMatch);
      div.appendChild(ex);

      frag.appendChild(div);
    });
    resultsEl.appendChild(frag);
  }

  function findMatch(matches, key) {
    for (var i = 0; i < matches.length; i++) {
      if (matches[i].key === key) return matches[i];
    }
    return null;
  }

  function renderTags(tags, tagsMatch) {
    if (!tags || !tags.length) return '';
    // 标签是数组。判断每个标签是否被命中:标签含某个搜索词,或某个搜索词含标签。
    return tags.map(function (t) {
      var isHit = false;
      for (var i = 0; i < _currentWords.length; i++) {
        var w = _currentWords[i];
        if (t.indexOf(w) !== -1 || w.indexOf(t) !== -1) { isHit = true; break; }
      }
      var inner = isHit ? '<mark>' + escapeHtml(t) + '</mark>' : escapeHtml(t);
      return '<span class="bs-result-tag">' + inner + '</span>';
    }).join('');
  }

  // 当前扩展后的搜索词集合(renderTags / queryHits 共用)
  var _currentWords = [];

  // —— 状态提示 ——
  function showStatus(msg, isError) {
    statusEl.textContent = msg;
    statusEl.className = 'blog-search-status' + (isError ? ' bs-error' : '');
  }
  function clearStatus() { statusEl.textContent = ''; statusEl.className = 'blog-search-status'; }

  // —— 主流程 ——
  var timer = null;
  function onInput() {
    var q = input.value.trim();
    clearTimeout(timer);
    if (!q) {
      // 清空:恢复文章列表
      document.body.classList.remove('bs-searching');
      resultsEl.innerHTML = '';
      clearStatus();
      return;
    }
    document.body.classList.add('bs-searching');
    showStatus('搜索中…', false);
    // 输入时防抖搜索
    timer = setTimeout(function () { searchNow(q); }, DEBOUNCE_MS);
  }

  // 立即搜索(回车触发,跳过防抖)
  function searchNow(q) {
    q = (q == null ? input.value.trim() : q).trim();
    if (!q) {
      document.body.classList.remove('bs-searching');
      resultsEl.innerHTML = '';
      clearStatus();
      return;
    }
    clearTimeout(timer);
    document.body.classList.add('bs-searching');
    if (!loaded) {
      showStatus('搜索中…', false);
      loadIndex(function () { if (loaded) runSearch(q); });
      return;
    }
    runSearch(q);
  }

  function runSearch(q) {
    _currentWords = expandQuery(q);
    var results = searchAll(q);
    render(results, q);
  }

  input.addEventListener('input', onInput);

  // 回车 / 点击搜索按钮:走 form 的 submit 事件(最可靠地覆盖两者)
  var form = document.getElementById('blog-search-form');
  if (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault(); // 阻止页面跳转
      searchNow(input.value);
    });
  }

  // 支持 URL ?q= 预填(为后续导航栏入口预留)
  try {
    var params = new URLSearchParams(window.location.search);
    var q = params.get('q');
    if (q) { input.value = q; onInput(); }
  } catch (e) { /* URLSearchParams 不可用时忽略 */ }

  // —— 工具 ——
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
})();
