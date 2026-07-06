(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // Local dev serves this file next to a locally-running API; a GitHub
  // Pages deployment has no local API to talk to, so it defaults to the
  // hosted Render backend instead. Either way, the gear icon overrides it.
  const DEPLOYED_API_BASE = "https://options-pricing-engine-api.onrender.com";
  const DEFAULT_API_BASE = location.hostname.endsWith("github.io") ? DEPLOYED_API_BASE : "http://127.0.0.1:8000";

  const state = {
    apiBase: localStorage.getItem("opx_api_base") || DEFAULT_API_BASE,
    side: "call",
  };

  // ---------------------------------------------------------------------
  // fetch helpers
  // ---------------------------------------------------------------------

  async function fetchJSON(path, params) {
    const url = new URL(state.apiBase.replace(/\/$/, "") + path);
    Object.entries(params || {}).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    });
    let res;
    try {
      res = await fetch(url);
    } catch (networkErr) {
      throw new Error(`Could not reach API at ${state.apiBase} (${networkErr.message}). Is the backend running?`);
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  function showBanner(msg, kind = "error") {
    const el = $("errorBanner");
    el.textContent = msg;
    el.classList.toggle("info", kind === "info");
    el.hidden = false;
  }
  function hideBanner() { $("errorBanner").hidden = true; }

  function setStatus(kind, text) {
    $("statusDot").className = "dot " + kind;
    $("statusText").textContent = text;
  }

  async function pingApi() {
    const isDeployedFree = state.apiBase === DEPLOYED_API_BASE;
    setStatus("pending", isDeployedFree ? "waking up backend… (free tier, ~30-50s if idle)" : "connecting…");
    try {
      await fetchJSON("/", {});
      setStatus("ok", "connected");
      hideBanner();
      return true;
    } catch (e) {
      setStatus("bad", "unreachable");
      showBanner(e.message);
      return false;
    }
  }

  // ---------------------------------------------------------------------
  // formatting
  // ---------------------------------------------------------------------

  const fmt = (x, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
  const fmtPct = (x, d = 2) => (x === null || x === undefined ? "—" : (Number(x) * 100).toFixed(d) + "%");
  const fmtInt = (x) => (x === null || x === undefined ? "—" : String(Math.round(x)));

  function modelLabel(model, mcVariant) {
    if (model === "bsm") return "BSM";
    if (model === "binomial") return "BINOMIAL (AMER)";
    if (model === "monte_carlo") return "MC · " + mcVariant.toUpperCase();
    return model.toUpperCase();
  }

  // ---------------------------------------------------------------------
  // expiries
  // ---------------------------------------------------------------------

  function pickDefaultExpiry(expiries) {
    const now = Date.now();
    const minMs = 7 * 24 * 3600 * 1000;
    for (const e of expiries) {
      if (new Date(e + "T21:00:00Z").getTime() - now >= minMs) return e;
    }
    return expiries[0];
  }

  async function loadExpiries(ticker) {
    const sel = $("expirySelect");
    sel.innerHTML = "<option>loading…</option>";
    try {
      const data = await fetchJSON("/expiries", { ticker });
      sel.innerHTML = "";
      data.expiries.forEach((e) => {
        const opt = document.createElement("option");
        opt.value = e; opt.textContent = e;
        sel.appendChild(opt);
      });
      sel.value = pickDefaultExpiry(data.expiries);
      hideBanner();
    } catch (e) {
      sel.innerHTML = "<option>—</option>";
      showBanner(e.message);
    }
  }

  // ---------------------------------------------------------------------
  // control strip wiring
  // ---------------------------------------------------------------------

  function wireControls() {
    $("sideToggle").querySelectorAll(".pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        $("sideToggle").querySelectorAll(".pill").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.side = btn.dataset.value;
      });
    });

    $("modelSelect").addEventListener("change", (e) => {
      const isMC = e.target.value === "monte_carlo";
      $("mcVariantField").hidden = !isMC;
      updateBarrierFields();
    });

    $("mcVariantSelect").addEventListener("change", updateBarrierFields);

    function updateBarrierFields() {
      const isMC = $("modelSelect").value === "monte_carlo";
      const isBarrier = isMC && $("mcVariantSelect").value === "barrier";
      $("barrierLevelField").hidden = !isBarrier;
      $("barrierTypeField").hidden = !isBarrier;
    }

    $("tickerInput").addEventListener("change", () => {
      const t = $("tickerInput").value.trim().toUpperCase();
      if (t) loadExpiries(t);
    });

    $("apiConfigBtn").addEventListener("click", () => {
      const panel = $("apiConfigPanel");
      panel.hidden = !panel.hidden;
      $("apiBaseInput").value = state.apiBase;
    });

    $("apiBaseSave").addEventListener("click", async () => {
      const val = $("apiBaseInput").value.trim();
      if (!val) return;
      state.apiBase = val;
      localStorage.setItem("opx_api_base", val);
      $("apiConfigPanel").hidden = true;
      const ok = await pingApi();
      if (ok) loadExpiries($("tickerInput").value.trim().toUpperCase());
    });

    $("priceBtn").addEventListener("click", handleSubmit);
  }

  // ---------------------------------------------------------------------
  // price + greeks
  // ---------------------------------------------------------------------

  async function handleSubmit() {
    const btn = $("priceBtn");
    btn.disabled = true;
    btn.textContent = "PRICING…";
    hideBanner();

    const ticker = $("tickerInput").value.trim().toUpperCase();
    const expiry = $("expirySelect").value;
    const strike = $("strikeInput").value;
    const model = $("modelSelect").value;
    const mcVariant = $("mcVariantSelect").value;
    const volSource = $("volSourceSelect").value;

    const baseParams = {
      ticker, expiry, strike: strike || undefined,
      option_type: state.side, vol_source: volSource,
    };

    try {
      const priceParams = { ...baseParams, model };
      if (model === "monte_carlo") {
        priceParams.mc_variant = mcVariant;
        if (mcVariant === "barrier") {
          priceParams.barrier = $("barrierLevelInput").value;
          priceParams.barrier_type = $("barrierTypeSelect").value;
        }
      }

      const pricePromise = fetchJSON("/price", priceParams);
      const greeksPromise = (model === "bsm" || model === "binomial")
        ? fetchJSON("/greeks", { ...baseParams, model })
        : Promise.resolve(null);
      const chainPromise = fetchJSON("/chain", { ticker, expiry, range_pct: 0.25 });
      const smilePromise = fetchJSON("/iv-smile", { ticker, expiry, range_pct: 0.25 });

      const [priceData, greeksData, chainData, smileData] = await Promise.all([
        pricePromise, greeksPromise, chainPromise, smilePromise,
      ]);

      renderPrice(priceData, model, mcVariant);
      renderGreeks(greeksData, model);
      renderChain(chainData);
      renderSmile(smileData);
    } catch (e) {
      showBanner(e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "PRICE IT";
    }
  }

  function renderPrice(data, model, mcVariant) {
    $("modelBadge").textContent = modelLabel(model, mcVariant);
    const priceStr = fmt(data.price, 4) + (data.std_error != null ? `  ± ${fmt(1.96 * data.std_error, 4)}` : "");
    $("priceHero").textContent = priceStr;
    $("sigmaLine").textContent =
      `σ ${fmtPct(data.sigma)} · ${data.sigma_source} · K=${fmt(data.strike, 2)} · T=${fmt(data.time_to_expiry_years, 4)}y · r=${fmtPct(data.risk_free_rate)} · q=${fmtPct(data.dividend_yield)}`;

    $("mktBid").textContent = fmt(data.market_bid, 2);
    $("mktAsk").textContent = fmt(data.market_ask, 2);
    $("mktMid").textContent = fmt(data.market_mid, 2);
    const diffEl = $("mktDiff");
    if (data.model_vs_market_diff == null) {
      diffEl.textContent = "—";
      diffEl.className = "mc-val";
    } else {
      diffEl.textContent = (data.model_vs_market_diff >= 0 ? "+" : "") + fmt(data.model_vs_market_diff, 4);
      diffEl.className = "mc-val " + (data.model_vs_market_diff >= 0 ? "pos" : "neg");
    }

    const notesEl = $("priceNotes");
    notesEl.innerHTML = "";
    (data.notes || []).forEach((n) => {
      const li = document.createElement("li");
      li.textContent = n;
      notesEl.appendChild(li);
    });
  }

  function renderGreeks(data, model) {
    $("greeksModelBadge").textContent = data ? modelLabel(model) : "N/A";
    const ids = ["gDelta", "gGamma", "gVega", "gTheta", "gRho"];
    const keys = ["delta", "gamma", "vega", "theta", "rho"];
    const note = $("greeksNote");
    if (!data) {
      ids.forEach((id) => ($(id).textContent = "—"));
      note.textContent = "Greeks are not implemented for Monte Carlo in this build (see README) — switch to BSM or Binomial.";
      return;
    }
    keys.forEach((k, i) => {
      const decimals = k === "gamma" ? 6 : 4;
      $(ids[i]).textContent = fmt(data[k], decimals);
    });
    note.textContent = model === "binomial"
      ? "Finite-difference (bump-and-reprice) off the tree — theta is known to be noisier than BSM's closed form."
      : "Closed-form analytic derivatives.";
  }

  // ---------------------------------------------------------------------
  // chain table
  // ---------------------------------------------------------------------

  function renderChain(data) {
    $("parityMeta").textContent =
      `${data.ticker} ${data.expiry} · parity: ${data.put_call_parity_violations}/${data.put_call_parity_checked} strikes violate tolerance`;

    const body = $("chainBody");
    body.innerHTML = "";
    const rows = [
      ...data.calls.map((r) => ({ ...r, side: "CALL", cls: "row-call" })),
      ...data.puts.map((r) => ({ ...r, side: "PUT", cls: "row-put" })),
    ];
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="9" class="chain-empty">No strikes in range.</td></tr>';
      return;
    }

    // yfinance's live bid/ask feed goes to $0.00 for every contract outside
    // trading hours (nights, weekends, holidays) even though volume/OI still
    // show real recent activity -- this is expected upstream behavior, not a
    // bug in the validators flagging every row CROSSED. Say so explicitly
    // instead of leaving an all-zero chain looking broken.
    const allZeroQuotes = rows.every((r) => (r.bid || 0) === 0 && (r.ask || 0) === 0);
    if (allZeroQuotes) {
      showBanner(
        "Market looks closed right now — yfinance is returning $0.00 bid/ask for every contract " +
        "(normal outside 9:30am–4:00pm ET, or on a market holiday). The validators are correctly " +
        "flagging these as CROSSED rather than pricing against a fake $0.00 quote. Prices above fall " +
        "back to realized vol; try again during market hours for live implied vol and a populated smile.",
        "info"
      );
    } else {
      hideBanner();
    }

    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.className = r.cls + (r.is_crossed || r.is_illiquid ? " row-flagged" : "");
      let flags = "";
      if (r.is_crossed) flags += '<span class="flag-tag flag-crossed">CROSSED</span>';
      if (r.is_illiquid) flags += '<span class="flag-tag flag-illiquid">ILLIQUID</span>';
      if (!flags) flags = '<span class="flag-ok">—</span>';
      tr.innerHTML = `
        <td class="col-side">${r.side}</td>
        <td>${fmt(r.strike, 2)}</td>
        <td>${fmt(r.bid, 2)}</td>
        <td>${fmt(r.ask, 2)}</td>
        <td>${fmt(r.mid, 2)}</td>
        <td>${fmtInt(r.volume)}</td>
        <td>${fmtInt(r.open_interest)}</td>
        <td>${fmtPct(r.iv)}</td>
        <td>${flags}</td>`;
      body.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------------
  // IV smile chart (hand-built SVG, no chart library)
  // ---------------------------------------------------------------------

  const SVG_NS = "http://www.w3.org/2000/svg";
  const svg = () => $("smileSvg");
  let currentSmile = null; // holds scales + refs for the mousemove handler

  function el(tag, attrs) {
    const e = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  }

  function renderSmile(data) {
    $("smileMeta").textContent = `${data.ticker} ${data.expiry} · spot ${fmt(data.spot, 2)}`;
    $("atmIvValue").textContent = fmtPct(data.atm_iv);

    const s = svg();
    s.innerHTML = "";
    currentSmile = null;

    const points = data.points || [];
    const usable = points.filter((p) => p.call_iv != null || p.put_iv != null);
    if (!usable.length) {
      $("chartEmpty").hidden = false;
      s.style.display = "none";
      return;
    }
    $("chartEmpty").hidden = true;
    s.style.display = "block";

    const strikes = points.map((p) => p.strike);
    const ivs = points.flatMap((p) => [p.call_iv, p.put_iv]).filter((v) => v != null);
    const xMin = Math.min(...strikes), xMax = Math.max(...strikes);
    const yMinRaw = Math.min(...ivs), yMaxRaw = Math.max(...ivs);
    const pad = Math.max((yMaxRaw - yMinRaw) * 0.15, 0.01);
    const yMin = Math.max(0, yMinRaw - pad), yMax = yMaxRaw + pad;

    const left = 54, right = 24, top = 16, bottom = 34;
    const W = 900, H = 340;
    const plotW = W - left - right, plotH = H - top - bottom;

    const xScale = (k) => left + ((k - xMin) / (xMax - xMin || 1)) * plotW;
    const yScale = (v) => top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

    // gridlines + y labels
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const v = yMin + ((yMax - yMin) * i) / ticks;
      const y = yScale(v);
      s.appendChild(el("line", { x1: left, x2: W - right, y1: y, y2: y, stroke: "#1a2230", "stroke-width": 1 }));
      const label = el("text", { x: left - 8, y: y + 4, "text-anchor": "end", fill: "#5b6472", "font-size": 10, "font-family": "monospace" });
      label.textContent = (v * 100).toFixed(0) + "%";
      s.appendChild(label);
    }

    // x labels (subset)
    const xTickCount = Math.min(8, strikes.length);
    const step = Math.max(1, Math.floor(strikes.length / xTickCount));
    strikes.forEach((k, i) => {
      if (i % step !== 0 && i !== strikes.length - 1) return;
      const x = xScale(k);
      const label = el("text", { x, y: H - bottom + 18, "text-anchor": "middle", fill: "#5b6472", "font-size": 10, "font-family": "monospace" });
      label.textContent = k % 1 === 0 ? String(k) : k.toFixed(1);
      s.appendChild(label);
    });

    // spot line
    if (data.spot >= xMin && data.spot <= xMax) {
      const x = xScale(data.spot);
      s.appendChild(el("line", { x1: x, x2: x, y1: top, y2: top + plotH, stroke: "#5b6472", "stroke-width": 1, "stroke-dasharray": "3,3" }));
      const label = el("text", { x: x + 4, y: top + 12, fill: "#8b96a5", "font-size": 10, "font-family": "monospace" });
      label.textContent = "SPOT";
      s.appendChild(label);
    }

    // ATM IV line
    if (data.atm_iv != null) {
      const y = yScale(data.atm_iv);
      s.appendChild(el("line", { x1: left, x2: W - right, y1: y, y2: y, stroke: "#f0a020", "stroke-width": 1, "stroke-dasharray": "5,4", opacity: 0.7 }));
    }

    function drawSeries(key, color) {
      const pts = points.filter((p) => p[key] != null);
      if (pts.length < 1) return;
      const polylinePoints = pts.map((p) => `${xScale(p.strike)},${yScale(p[key])}`).join(" ");
      if (pts.length > 1) {
        s.appendChild(el("polyline", {
          points: polylinePoints, fill: "none", stroke: color,
          "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round",
        }));
      }
      pts.forEach((p) => {
        s.appendChild(el("circle", { cx: xScale(p.strike), cy: yScale(p[key]), r: 3, fill: color }));
      });
    }

    drawSeries("call_iv", "#1fa971");
    drawSeries("put_iv", "#e5484d");

    // interactive overlay: crosshair + highlight dots, updated on mousemove
    const crosshair = el("line", { x1: 0, x2: 0, y1: top, y2: top + plotH, stroke: "#e8edf3", "stroke-width": 1, opacity: 0 });
    const dotCall = el("circle", { r: 4.5, fill: "#1fa971", stroke: "#0a0e14", "stroke-width": 1.5, opacity: 0 });
    const dotPut = el("circle", { r: 4.5, fill: "#e5484d", stroke: "#0a0e14", "stroke-width": 1.5, opacity: 0 });
    s.appendChild(crosshair);
    s.appendChild(dotCall);
    s.appendChild(dotPut);

    const overlay = el("rect", { x: left, y: top, width: plotW, height: plotH, fill: "transparent", style: "cursor:crosshair" });
    s.appendChild(overlay);

    currentSmile = { points, xMin, xMax, left, plotW, xScale, yScale, crosshair, dotCall, dotPut };
  }

  function handleSmileMouseMove(evt) {
    if (!currentSmile) return;
    const s = svg();
    const pt = s.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const ctm = s.getScreenCTM();
    if (!ctm) return;
    const loc = pt.matrixTransform(ctm.inverse());

    const { points, xMin, xMax, left, plotW, xScale, yScale, crosshair, dotCall, dotPut } = currentSmile;
    const clampedX = Math.min(Math.max(loc.x, left), left + plotW);
    const strikeAtX = xMin + ((clampedX - left) / plotW) * (xMax - xMin);
    let nearest = points[0], bestDist = Infinity;
    for (const p of points) {
      const d = Math.abs(p.strike - strikeAtX);
      if (d < bestDist) { bestDist = d; nearest = p; }
    }

    const x = xScale(nearest.strike);
    crosshair.setAttribute("x1", x); crosshair.setAttribute("x2", x); crosshair.setAttribute("opacity", 0.5);

    if (nearest.call_iv != null) {
      dotCall.setAttribute("cx", x); dotCall.setAttribute("cy", yScale(nearest.call_iv)); dotCall.setAttribute("opacity", 1);
    } else dotCall.setAttribute("opacity", 0);

    if (nearest.put_iv != null) {
      dotPut.setAttribute("cx", x); dotPut.setAttribute("cy", yScale(nearest.put_iv)); dotPut.setAttribute("opacity", 1);
    } else dotPut.setAttribute("opacity", 0);

    const tooltip = $("chartTooltip");
    tooltip.innerHTML = `
      <div class="tt-strike">STRIKE ${fmt(nearest.strike, 2)}</div>
      <div class="tt-row"><span class="tt-call">CALL IV</span><span>${fmtPct(nearest.call_iv)}</span></div>
      <div class="tt-row"><span class="tt-put">PUT IV</span><span>${fmtPct(nearest.put_iv)}</span></div>`;
    tooltip.hidden = false;

    const wrapRect = $("chartWrap").getBoundingClientRect();
    const left_px = evt.clientX - wrapRect.left + 14;
    const top_px = evt.clientY - wrapRect.top - 10;
    tooltip.style.left = Math.min(left_px, wrapRect.width - 150) + "px";
    tooltip.style.top = Math.max(top_px, 0) + "px";
  }

  function handleSmileMouseLeave() {
    if (currentSmile) currentSmile.crosshair.setAttribute("opacity", 0);
    if (currentSmile) { currentSmile.dotCall.setAttribute("opacity", 0); currentSmile.dotPut.setAttribute("opacity", 0); }
    $("chartTooltip").hidden = true;
  }

  // ---------------------------------------------------------------------
  // init
  // ---------------------------------------------------------------------

  function init() {
    wireControls();
    svg().addEventListener("mousemove", handleSmileMouseMove);
    svg().addEventListener("mouseleave", handleSmileMouseLeave);
    pingApi().then((ok) => { if (ok) loadExpiries($("tickerInput").value.trim().toUpperCase()); });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
