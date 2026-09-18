/**
 * file_path: backend/static/js/overlay.js
 * 영상 위에 검출 박스 또는 보행가능(초록색)·횡단보도(핑크색) 마스크를 표시한다.
 * object-fit: contain 여백을 보정하고 종료 후 늦게 도착한 마스크는 무시한다.
 */
window.GOverlay = (() => {
  const canvas = document.getElementById("overlay");
  const video = document.getElementById("video");
  const ctx = canvas.getContext("2d");
  const COLORS = ["#4f8cff", "#34c759", "#ffb020", "#ff4d4f", "#b57bff", "#22c9c9"];
  let drawVersion = 0;

  function describeDetection(d) {
    const state = d.class_name === "pedestrian_signal" ? d.extra?.signal_state : null;
    if (state === "red" || state === "green" || state === "unknown") {
      const label = { red: "빨간불", green: "초록불", unknown: "신호 미확인" }[state];
      const score = d.extra?.color_confidence;
      return {
        label,
        confidenceText: state !== "unknown" && typeof score === "number" && Number.isFinite(score)
          ? `${(score * 100).toFixed(1)}%` : "",
        color: { red: "#e5393b", green: "#1fa84a", unknown: "#888888" }[state],
      };
    }
    return {
      label: d.class_name,
      confidenceText: `${(d.confidence * 100).toFixed(0)}%`,
      color: COLORS[(d.class_id ?? 0) % COLORS.length],
    };
  }

  function fit() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  // contain 으로 표시된 실제 영상 영역
  function contentRect(w, h) {
    const vw = video.videoWidth || 3, vh = video.videoHeight || 4;
    const s = Math.min(w / vw, h / vh);
    const cw = vw * s, ch = vh * s;
    return { x: (w - cw) / 2, y: (h - ch) / 2, w: cw, h: ch };
  }

  // 이전 마스크의 비동기 표시 취소
  /** 화면을 지우고 아직 디코딩 중인 마스크를 무효화한다. */
  function clear() {
    drawVersion += 1;
    const { w, h } = fit();
    ctx.clearRect(0, 0, w, h);
  }

  // 최신 프레임의 추론 결과 표시
  /** 박스를 그리거나 PNG 디코딩 후 최신 요청의 초록색·핑크색 마스크를 그린다. */
  function draw(detections, event = {}) {
    const version = ++drawVersion;
    if (event.mask_png) {
      const mask = new Image();
      mask.onload = () => {
        if (version !== drawVersion) return;
        const { w, h } = fit();
        const r = contentRect(w, h);
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(mask, r.x, r.y, r.w, r.h);
      };
      mask.onerror = () => { if (version === drawVersion) clear(); };
      mask.src = `data:image/png;base64,${event.mask_png}`;
      return;
    }
    const { w, h } = fit();
    ctx.clearRect(0, 0, w, h);
    const r = contentRect(w, h);
    ctx.lineWidth = 2.5;
    ctx.font = "600 13px -apple-system, Roboto, sans-serif";
    ctx.textBaseline = "top";
    for (const d of detections || []) {
      const b = d.box;
      const x1 = r.x + b.x1 * r.w, y1 = r.y + b.y1 * r.h;
      const x2 = r.x + b.x2 * r.w, y2 = r.y + b.y2 * r.h;
      const display = describeDetection(d);
      const color = display.color;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${display.label}${display.confidenceText ? ` ${display.confidenceText}` : ""}`
        + (d.track_id != null ? ` #${d.track_id}` : "");
      const tw = ctx.measureText(label).width + 10, th = 18;
      const ly = y1 - th < r.y ? y1 : y1 - th;
      ctx.fillStyle = color;
      ctx.fillRect(x1, ly, tw, th);
      ctx.fillStyle = "#0b0d12";
      ctx.fillText(label, x1 + 5, ly + 2);
    }
  }

  window.addEventListener("resize", () => fit());
  return { draw, clear, describeDetection };
})();
