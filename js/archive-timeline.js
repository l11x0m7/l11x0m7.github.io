/* ============================================================
   Archive — Cyberpunk Timeline 滚动逐个点亮
   原生 IntersectionObserver,不依赖 jQuery。
   启动时给 .cyber-timeline 加 .js 类启用 reveal 初始态;
   视口内元素同步点亮(避免首屏闪烁),其余滚动进入时点亮。
   ============================================================ */
(function () {
  var timeline = document.querySelector('.cyber-timeline');
  if (!timeline) return;

  var items = timeline.querySelectorAll('.tl-item');
  // 启用 reveal 初始态(opacity:0),无 JS 时卡片默认可见
  timeline.classList.add('js');

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
    return;
  }

  // 先同步点亮当前已在视口内的(避免首屏闪烁)
  Array.prototype.forEach.call(items, function (el, i) {
    if (inViewport(el)) reveal(el, Math.min(i % 10, 8) * 50);
  });

  var io = new IntersectionObserver(function (entries) {
    Array.prototype.forEach.call(entries, function (entry) {
      if (entry.isIntersecting) {
        var el = entry.target;
        var idx = Array.prototype.indexOf.call(items, el);
        reveal(el, Math.min(idx % 10, 8) * 50);
        io.unobserve(el);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -10% 0px' });

  Array.prototype.forEach.call(items, function (el) {
    if (!el.classList.contains('in-view')) io.observe(el);
  });
})();
