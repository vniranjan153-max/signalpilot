(async function () {
  const $ = (s, r = document) => r.querySelector(s);
  const fmt = (n, d = 2) => n == null ? "n/a" : Number(n).toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d});
  const pct = (n, d = 1) => n == null ? "n/a" : (n * 100).toFixed(d) + "%";
  const cls = n => n == null ? "" : n >= 0 ? "pos" : "neg";
  let data;
  try {
    const r = await fetch("data/signals.json?v=" + Date.now());
    if (!r.ok) throw new Error(r.status);
    data = await r.json();
  } catch (e) {
    $("#meta").textContent = "Could not load data/signals.json. Run the pipeline first (see README).";
    return;
  }
  $("#disc").textContent = data.disclaimer;
  $("#hz").textContent = data.horizon_days;
  if (data.mode === "demo") $("#demo").hidden = false;
  const when = new Date(data.generated_at);
  $("#meta").textContent = `Updated ${when.toLocaleString()} · ${data.assets.length} markets · ${data.horizon_days}-day horizon · ${data.mode} data` +
    (data.errors.length ? ` · ${data.errors.length} failed` : "");

  function quality(auc) {
    if (auc == null) return ["n/a", "none"];
    if (auc >= 0.56) return ["Some edge · AUC " + auc.toFixed(2), "good"];
    if (auc >= 0.52) return ["Weak edge · AUC " + auc.toFixed(2), "weakq"];
    return ["No edge · AUC " + auc.toFixed(2), "none"];
  }

  function draw(canvas, series, colors) {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const g = canvas.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const all = series.flat().filter(v => v != null);
    const lo = Math.min(...all), hi = Math.max(...all), pad = 6;
    series.forEach((s, k) => {
      g.beginPath(); g.lineWidth = k === 0 ? 2 : 1.4; g.strokeStyle = colors[k];
      s.forEach((v, i) => {
        const x = (i / (s.length - 1)) * w, y = h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
    });
  }

  function card(a) {
    const [qtxt, qcls] = quality(a.metrics.auc);
    const m = a.metrics, el = document.createElement("article");
    el.className = "card" + (qcls === "none" ? " weak" : ""); el.dataset.signal = a.signal;
    el.innerHTML = `
      <div class="row"><div><div class="name">${a.name}</div><div class="sym">${a.symbol} · as of ${a.asof}</div></div>
        <span class="badge ${a.signal}">${a.signal}</span></div>
      <div class="row"><div class="price">${fmt(a.close)}<small>last close</small></div><span class="q ${qcls}">${qtxt}</span></div>
      <div><div class="gauge" aria-label="Probability up ${pct(a.prob_up, 0)}"><i style="left:${a.prob_up * 100}%"></i></div>
        <div class="gl"><span>Down</span><span><b>P(up) ${pct(a.prob_up, 0)}</b></span><span>Up</span></div></div>
      ${a.levels ? `<div class="lv">Illustrative risk levels (ATR): stop ${fmt(a.levels.stop)} · target ${fmt(a.levels.target)}</div>` : ""}
      <div class="tabs"><button class="tab on" data-t="px">Price</button><button class="tab" data-t="eq">Backtest equity</button></div>
      <canvas></canvas>
      <div class="stats">
        <div class="stat"><b class="${cls(m.strategy_cagr)}">${pct(m.strategy_cagr)}</b><span>Strategy / yr</span></div>
        <div class="stat"><b class="${cls(m.buyhold_cagr)}">${pct(m.buyhold_cagr)}</b><span>Buy &amp; hold / yr</span></div>
        <div class="stat"><b>${fmt(m.sharpe)}</b><span>Sharpe</span></div>
        <div class="stat"><b class="neg">${pct(m.max_drawdown)}</b><span>Max drawdown</span></div>
        <div class="stat"><b>${m.trades}</b><span>Trades</span></div>
        <div class="stat"><b>${pct(m.time_in_market, 0)}</b><span>Time invested</span></div>
      </div>`;
    const cv = $("canvas", el), show = t => {
      el.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.t === t));
      t === "px" ? draw(cv, [a.history.close], ["#6ea8ff"]) : draw(cv, [a.equity.strategy, a.equity.buyhold], ["#2ecc71", "#8d99ae"]);
    };
    el.querySelectorAll(".tab").forEach(b => b.onclick = () => show(b.dataset.t));
    el._draw = () => show(el.querySelector(".tab.on").dataset.t);
    return el;
  }

  const grid = $("#grid"), cards = data.assets.map(card);
  cards.forEach(c => grid.appendChild(c));
  const redraw = () => cards.forEach(c => c.hidden || c._draw());
  redraw(); addEventListener("resize", redraw);
  document.querySelectorAll(".chip").forEach(ch => ch.onclick = () => {
    document.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === ch));
    cards.forEach(c => c.hidden = ch.dataset.f !== "ALL" && c.dataset.signal !== ch.dataset.f);
    redraw();
  });
  data.errors.forEach(e => { const d = document.createElement("div"); d.className = "banner warn"; d.textContent = `${e.symbol}: ${e.error}`; grid.after(d); });
})();
