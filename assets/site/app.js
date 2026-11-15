/* Interacciones del artículo: progreso de lectura, revelado al hacer scroll,
   índice activo y botón para volver arriba. Todo respeta prefers-reduced-motion. */

(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------- progreso */
  var bar = document.querySelector(".progress");
  var topbar = document.querySelector(".topbar");
  var totop = document.querySelector(".totop");
  var now = document.querySelector(".topbar .now b");

  var sections = Array.prototype.slice.call(
    document.querySelectorAll("section.chapter[id]")
  );

  function onScroll() {
    var y = window.scrollY || document.documentElement.scrollTop;
    var h = document.documentElement.scrollHeight - window.innerHeight;
    var pct = h > 0 ? Math.min(100, Math.max(0, (y / h) * 100)) : 0;

    if (bar) bar.style.width = pct.toFixed(2) + "%";
    if (topbar) topbar.classList.toggle("scrolled", y > 12);
    if (totop) totop.classList.toggle("show", y > window.innerHeight * 0.6);

    var current = null;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].getBoundingClientRect().top <= 140) current = sections[i];
    }
    if (current && now) now.textContent = current.dataset.short || "";

    if (!reduced) {
      ticking = false;
    }
  }

  var ticking = false;
  window.addEventListener(
    "scroll",
    function () {
      if (!ticking) {
        window.requestAnimationFrame(onScroll);
        ticking = true;
      }
    },
    { passive: true }
  );

  /* ------------------------------------------------------- revelado */
  var revealables = Array.prototype.slice.call(
    document.querySelectorAll(".reveal")
  );

  if (reduced || !("IntersectionObserver" in window)) {
    revealables.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            io.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.06 }
    );
    revealables.forEach(function (el) { io.observe(el); });
  }

  /* ------------------------------------------------------- volver arriba */
  if (totop) {
    totop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
  }

  /* ------------------------------------------------------- scroll suave con offset */
  document.querySelectorAll('a[href^="#"]').forEach(function (link) {
    link.addEventListener("click", function (event) {
      var id = link.getAttribute("href").slice(1);
      var target = document.getElementById(id);
      if (!target) return;
      event.preventDefault();
      var top = target.getBoundingClientRect().top + window.scrollY - 86;
      window.scrollTo({ top: top, behavior: reduced ? "auto" : "smooth" });
      history.replaceState(null, "", "#" + id);
    });
  });

  onScroll();
})();
