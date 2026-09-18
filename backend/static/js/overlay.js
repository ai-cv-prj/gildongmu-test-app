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
  let pendingDraw = null;
  let maskCanvas = null, maskContext = null, maskPixels = null;
  // RGBA 바이트 배열을 같은 플랫폼의 uint32로 읽어 엔디언에 무관하게 채운다.
  const MASK_COLORS = new Uint32Array(new Uint8Array([
    0, 0, 0, 0, 0, 255, 0, 140, 255, 105, 180, 140,
  ]).buffer);

  // 비동기 그리기 측정 완료
  /** 실제 canvas 그리기 완료 또는 취소를 호출자에게 알린다. */
  function finishDraw(status) {
    const resolve = pendingDraw;
    pendingDraw = null;
    if (resolve) resolve({ status, drawn_ms: status === "drawn" ? performance.now() : null });
  }

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
  /** 종료·오류 상태를 알리고 화면 및 아직 디코딩 중인 마스크를 정리한다. */
  function clear(status = "superseded") {
    finishDraw(status);
    drawVersion += 1;
    const { w, h } = fit();
    ctx.clearRect(0, 0, w, h);
  }

  // 디코딩된 최신 마스크 표시
  /** 취소된 결과는 무시하고 실제로 canvas에 그린 시각을 기록한다. */
  function paintMask(mask, version) {
    if (version !== drawVersion) return;
    try {
      const { w, h } = fit();
      const r = contentRect(w, h);
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(mask, r.x, r.y, r.w, r.h);
      finishDraw("drawn");
    } catch (_) {
      clear("error");
    }
  }

  // 구형 브라우저의 PNG 표시 경로
  /** ImageBitmap을 지원하지 않거나 디코딩에 실패하면 기존 Image 방식을 사용한다. */
  function loadMaskImage(encoded, version) {
    if (version !== drawVersion) return;
    const mask = new Image();
    mask.onload = () => paintMask(mask, version);
    mask.onerror = () => {
      if (version === drawVersion) clear("error");
    };
    mask.src = `data:image/png;base64,${encoded}`;
  }

  // DOM 이미지 로드 단계를 거치지 않는 PNG 디코딩
  /** PNG를 직접 ImageBitmap으로 디코딩하며 사용한 비트맵 자원을 해제한다. */
  async function decodeMask(encoded, version) {
    let bitmap;
    try {
      const raw = atob(encoded);
      const bytes = Uint8Array.from(raw, (character) => character.charCodeAt(0));
      bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
      paintMask(bitmap, version);
    } catch (_) {
      loadMaskImage(encoded, version);
    } finally {
      bitmap?.close();
    }
  }

  // PNG 이미지 디코딩 대기 없이 마스크 복원
  /** 길이·색상 구간을 원래 RGBA 픽셀로 복원하고 기존 영상 영역에 그린다. */
  function drawRleMask(mask, version) {
    try {
      const { width, height, data } = mask;
      const count = width * height;
      if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0
          || count > 4194304 || typeof data !== "string" || data.length > 21848) {
        throw new Error("잘못된 마스크 크기");
      }
      const raw = atob(data);
      if (!raw.length || raw.length % 4) throw new Error("잘못된 마스크 데이터");
      const bytes = Uint8Array.from(raw, (character) => character.charCodeAt(0));
      const runs = new DataView(bytes.buffer);
      if (!maskCanvas) {
        maskCanvas = document.createElement("canvas");
        maskContext = maskCanvas.getContext("2d");
      }
      if (!maskPixels || maskCanvas.width !== width || maskCanvas.height !== height) {
        maskCanvas.width = width; maskCanvas.height = height;
        maskPixels = maskContext.createImageData(width, height);
      }
      const pixels = new Uint32Array(maskPixels.data.buffer);
      let cursor = 0;
      for (let offset = 0; offset < bytes.length; offset += 4) {
        const run = runs.getUint32(offset, true), color = run & 3, length = run >>> 2;
        if (!length || color > 2 || cursor + length > count) throw new Error("잘못된 마스크 구간");
        pixels.fill(MASK_COLORS[color], cursor, cursor + length);
        cursor += length;
      }
      if (cursor !== count) throw new Error("마스크 픽셀 누락");
      maskContext.putImageData(maskPixels, 0, 0);
      paintMask(maskCanvas, version);
    } catch (_) {
      if (version === drawVersion) clear("error");
    }
  }

  // 최신 프레임의 추론 결과 표시
  /** 박스·마스크를 그리고 그리기 완료 시각을 Promise로 반환한다. */
  function draw(detections, event = {}) {
    finishDraw("superseded");
    const completion = new Promise((resolve) => { pendingDraw = resolve; });
    const version = ++drawVersion;
    if (event.mask_rle) {
      drawRleMask(event.mask_rle, version);
      return completion;
    }
    if (event.mask_png) {
      if (window.createImageBitmap) void decodeMask(event.mask_png, version);
      else loadMaskImage(event.mask_png, version);
      return completion;
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
    finishDraw("drawn");
    return completion;
  }

  window.addEventListener("resize", () => fit());
  return { draw, clear, describeDetection };
})();
