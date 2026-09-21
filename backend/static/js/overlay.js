// 정규화 좌표(0~1) 검출 박스를 video 위 canvas 에 그린다. object-fit: contain 여백을 보정한다.
window.GOverlay = (() => {
  const canvas = document.getElementById("overlay");
  const video = document.getElementById("video");
  const ctx = canvas.getContext("2d");
  const COLORS = ["#4f8cff", "#34c759", "#ffb020", "#ff4d4f", "#b57bff", "#22c9c9"];

  function describeDetection(d) {
    if (d.class_name === "crosswalk") {
      const status = d.extra?.crosswalk_status;
      const detail = { below_confidence: "신뢰도 미달", position_rejected: "위치 조건 탈락",
        eligible: "검출", used: "연결 판단에 사용" }[status] || "검출";
      return { label: `횡단보도 · ${detail}`, confidenceText: `${(d.confidence * 100).toFixed(1)}%`,
        color: status === "used" ? "#22c9c9" : "#b57bff" };
    }
    const selection = d.class_name === "pedestrian_signal" ? d.extra?.selection_status : null;
    if (selection === "unselected" || selection === "candidate") {
      return {
        label: selection === "candidate" ? "신호등 · 선택 확인 중" : "신호등 · 미선택",
        confidenceText: `검출 ${(d.confidence * 100).toFixed(0)}%`,
        color: selection === "candidate" ? "#ffb020" : "#4f8cff",
      };
    }
    const state = d.class_name === "pedestrian_signal" ? d.extra?.signal_state : null;
    if (state === "red" || state === "green" || state === "unknown") {
      const label = { red: "빨간불", green: "초록불", unknown: "신호 미확인" }[state];
      const score = d.extra?.color_confidence;
      return {
        label: selection === "selected" ? `안내 대상 · ${label}` : label,
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

  function describeCrosswalkEvent(event) {
    const info = event.crosswalk_diagnostics;
    if (!info) return ""; // 기존 기록과 mock 응답에는 진단 정보가 없다.
    const detection = { not_detected: "횡단보도 검출 없음", below_confidence: "연결 신뢰도 기준 미달",
      position_rejected: "위치 조건 탈락", eligible: "위치 조건 통과" }[info.detection_status] || "검출";
    const connection = {
      ambiguous_crosswalks: "연결할 횡단보도 선택 불가",
      vanishing_point_unavailable: "횡단보도 방향 확인 불가",
      no_signal_in_crossing_direction: "횡단보도 방향의 신호등 없음",
      ambiguous_signals: "연결할 신호등 선택 불가",
      waiting_for_temporal_consistency: "신호등 연결 확인 중",
      waiting_for_target_switch: "다른 신호등으로 변경 확인 중 · 색상 안내 보류",
      waiting_for_target_revalidation: "기존 신호등 연결 재확인 중 · 색상 안내 보류",
      target_switched: "횡단보도 연결 확인 · 신호등 대상 변경",
      target_revalidated: "기존 신호등 연결 재확인",
      matched: "신호등 연결 확인",
      previous_target_retained: "기존 신호등 추적 중",
      crosswalk_relation_unverified: "단일 신호등 판별 · 횡단보도 연결 미확인",
      no_signal_detected: "연결할 신호등 없음",
    }[info.connection_status];
    const parts = info.candidate_count ? [`횡단보도 ${info.candidate_count}개`, detection] : [detection];
    if (connection) parts.push(connection);
    return parts.join(" · ");
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

  function clear() {
    const { w, h } = fit();
    ctx.clearRect(0, 0, w, h);
  }

  function draw(detections) {
    const { w, h } = fit();
    ctx.clearRect(0, 0, w, h);
    const r = contentRect(w, h);
    ctx.lineWidth = 2.5;
    ctx.font = "600 13px -apple-system, Roboto, sans-serif";
    ctx.textBaseline = "top";
    // 큰 횡단보도 박스 위에 작은 신호등 박스를 그린다. 응답 순서는 변경하지 않는다.
    const ordered = [...(detections || []).filter(d => d.class_name === "crosswalk"),
      ...(detections || []).filter(d => d.class_name !== "crosswalk")];
    const labelAreas = [];
    const labels = [];
    for (const d of ordered) {
      const b = d.box;
      const x1 = r.x + b.x1 * r.w, y1 = r.y + b.y1 * r.h;
      const x2 = r.x + b.x2 * r.w, y2 = r.y + b.y2 * r.h;
      const display = describeDetection(d);
      const color = display.color;
      const selection = d.extra?.selection_status;
      ctx.lineWidth = selection === "selected" ? 4 : selection ? 2 : 2.5;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${display.label}${display.confidenceText ? ` ${display.confidenceText}` : ""}`
        + (d.track_id != null ? ` #${d.track_id}` : "");
      const tw = Math.min(ctx.measureText(label).width + 10, r.w), th = 18;
      const preferredY = y1 - th < r.y ? y1 : y1 - th;
      const lx = Math.max(r.x, Math.min(x1, r.x + r.w - tw));
      const candidateYs = [preferredY, ...labelAreas.flatMap(a => [a.y - th - 2, a.y + th + 2])];
      const ly = candidateYs.find(y => y >= r.y && y + th <= r.y + r.h
        && labelAreas.every(a => lx + tw <= a.x || lx >= a.x + a.w || y + th <= a.y || y >= a.y + th))
        ?? Math.max(r.y, Math.min(preferredY, r.y + r.h - th));
      labelAreas.push({ x: lx, y: ly, w: tw });
      labels.push({ label, color, lx, ly, tw, th });
    }
    // 다른 박스의 테두리가 이미 그린 라벨을 덮지 않도록 라벨은 마지막에 그린다.
    for (const { label, color, lx, ly, tw, th } of labels) {
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, tw, th);
      ctx.fillStyle = "#0b0d12";
      ctx.fillText(label, lx + 5, ly + 2, Math.max(1, tw - 10));
    }
  }

  window.addEventListener("resize", () => fit());
  return { draw, clear, describeDetection, describeCrosswalkEvent };
})();
