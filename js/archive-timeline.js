/* ============================================================
   Archive — Cyberpunk Timeline 滚动逐个点亮
   原生 IntersectionObserver,不依赖 jQuery。
   启动时给 .cyber-timeline 加 .js 类启用 reveal 初始态;
   视口内元素同步点亮(避免首屏闪烁),其余滚动进入时点亮。
   ============================================================ */
(function () {
  // 节点间距按相邻两篇发文的真实天数间隔换算(平方根压缩)。
  // gapPx = clamp(MIN_GAP, sqrt(days) * GAP_SCALE, MAX_GAP)
  var GAP_SCALE = 14;   // 1 天 ≈ 14px;16 天 ≈ 56px;100 天 ≈ 140px
  var MIN_GAP = 12;     // 同日多篇(days=0)的最小间距,防重叠
  var MAX_GAP = 160;    // 封顶,防止数月间隔把页面拉得过长
  var DAY_MS = 86400000;

  var timeline = document.querySelector('.cyber-timeline');
  if (!timeline) return;

  var items = timeline.querySelectorAll('.tl-item');
  var yearEls = timeline.querySelectorAll('.tl-year');
  // 启用 reveal 初始态(opacity:0),无 JS 时卡片默认可见
  timeline.classList.add('js');

  // 把每个年份标签里的数字拆成单独 <span class="tl-digit">,用于逐字错相动画。
  // 例如 2017 → <span class="tl-digit" style="--i:0">2</span>...
  Array.prototype.forEach.call(yearEls, function (year) {
    var label = year.querySelector('.tl-year-label');
    if (!label) return;
    var text = label.textContent.trim();
    label.textContent = '';
    Array.prototype.forEach.call(text, function (ch, i) {
      var s = document.createElement('span');
      s.className = 'tl-digit';
      s.style.setProperty('--i', i);
      s.textContent = ch;
      label.appendChild(s);
    });
  });

  // 按相邻两篇发文的真实时间间隔换算节点间距(平方根压缩)。
  // DOM 顺序即 site.posts 倒序(最新在上):items[i] 比 items[i+1] 更新,
  // 间隔取绝对天数差。从第二个 item 起,把算出的间距写入 inline margin-top;
  // 第一个 item 紧跟年份标签,沿用默认上边距。年份标签不参与换算(仍是分组锚点)。
  if (items.length > 1) {
    // 解析每个 item 的发文日期(来自 .tl-date 的 datetime 属性)
    var dates = Array.prototype.map.call(items, function (el) {
      var t = el.querySelector('.tl-date');
      var d = t && t.getAttribute('datetime');
      var ms = d ? Date.parse(d) : NaN;
      return isNaN(ms) ? null : ms;
    });
    for (var i = 1; i < items.length; i++) {
      var prev = dates[i - 1];
      var curr = dates[i];
      if (prev == null || curr == null) continue; // 日期缺失则跳过,保留默认间距
      var days = Math.abs(prev - curr) / DAY_MS;
      var gap = Math.min(Math.max(Math.sqrt(days) * GAP_SCALE, MIN_GAP), MAX_GAP);
      items[i].style.marginTop = Math.round(gap) + 'px';
    }
  }

  function reveal(el, delay) {
    el.style.transitionDelay = delay + 'ms';
    el.classList.add('in-view');
  }

  function inViewport(el) {
    var r = el.getBoundingClientRect();
    var vh = window.innerHeight || document.documentElement.clientHeight;
    return r.top < vh * 0.88 && r.bottom > vh * 0.12;
  }

  // 不支持 IntersectionObserver:全部直接点亮
  if (!('IntersectionObserver' in window)) {
    Array.prototype.forEach.call(items, function (el, i) {
      reveal(el, Math.min(i % 10, 8) * 50);
    });
    Array.prototype.forEach.call(yearEls, function (el) {
      el.classList.add('in-view');
    });
    return;
  }

  // 先同步点亮当前已在视口内的(避免首屏闪烁)
  Array.prototype.forEach.call(items, function (el, i) {
    if (inViewport(el)) reveal(el, Math.min(i % 10, 8) * 50);
  });
  // 年份同样先同步点亮已在视口内的
  Array.prototype.forEach.call(yearEls, function (el) {
    if (inViewport(el)) el.classList.add('in-view');
  });

  var io = new IntersectionObserver(function (entries) {
    Array.prototype.forEach.call(entries, function (entry) {
      if (entry.isIntersecting) {
        var el = entry.target;
        if (el.classList.contains('tl-item')) {
          var idx = Array.prototype.indexOf.call(items, el);
          reveal(el, Math.min(idx % 10, 8) * 50);
        } else {
          // .tl-year 进入视口:触发逐字浮现
          el.classList.add('in-view');
        }
        io.unobserve(el);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -10% 0px' });

  Array.prototype.forEach.call(items, function (el) {
    if (!el.classList.contains('in-view')) io.observe(el);
  });
  Array.prototype.forEach.call(yearEls, function (el) {
    if (!el.classList.contains('in-view')) io.observe(el);
  });
})();
