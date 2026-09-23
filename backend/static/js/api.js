/**
 * 파일 경로: backend/static/js/api.js
 * 프레임·실시간 녹화를 전송하고 세션별 지연 로그를 모아 저장·재시도한다.
 */
window.GApi = (() => {
  const BASE = "/api";

  class ApiError extends Error {
    constructor(status, code, message, detail) {
      super(message);
      this.status = status;
      this.code = code;
      this.detail = detail;
    }
  }

  async function parse(res) {
    let body = null;
    try { body = await res.json(); } catch (_) { /* JSON 응답 본문을 읽을 수 없으면 기본값 유지 */ }
    if (!res.ok) {
      const err = (body && body.error) || {};
      throw new ApiError(res.status, err.code || "http_error", err.message || `HTTP ${res.status}`, err.detail);
    }
    return body;
  }

  async function request(path, opts = {}, timeoutMs = 8000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(BASE + path, { ...opts, signal: ctrl.signal, cache: "no-store" });
      return await parse(res);
    } catch (e) {
      if (e.name === "AbortError") throw new ApiError(0, "timeout", "요청 시간 초과");
      if (e instanceof ApiError) throw e;
      throw new ApiError(0, "network", "서버에 연결할 수 없습니다");
    } finally {
      clearTimeout(timer);
    }
  }

  const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  // 세션 단위 로그 수집기 생성
  /** 프레임 처리를 막지 않고 2.5초마다 최대 25건을 전송하며 실패한 묶음은 재시도한다. */
  function createTimings(sessionId, onWarning = () => {}) {
    const queue = [];
    const pending = new Set();
    let batch = null, sending = null, dropped = 0, closing = false;
    let lastDrawn = null, lastCapture = null;
    const timer = setInterval(() => { void flush(); }, 2500);

    // 완료된 측정값을 서버에 전송
    /** 실패 시 같은 batch_id를 유지하여 재전송 여부를 구별할 수 있게 한다. */
    function flush() {
      if (sending) return sending;
      if (!batch && !queue.length) return Promise.resolve(true);
      if (!batch) {
        batch = { batch_id: crypto.randomUUID(), dropped_records: dropped, records: queue.splice(0, 25) };
      }
      sending = (async () => {
        try {
          await request(`/sessions/${encodeURIComponent(sessionId)}/client-timings`, json(batch), 5000);
          batch = null;
          onWarning(dropped ? `지연 로그 ${dropped}건이 버퍼 초과로 누락되었습니다.` : "");
          return true;
        } catch (error) {
          onWarning(`지연 로그 저장 실패: ${error.message}. 페이지를 닫지 않으면 재시도합니다.`);
          return false;
        } finally {
          sending = null;
          if (closing && !batch && !queue.length && !pending.size) clearInterval(timer);
        }
      })();
      return sending;
    }

    // 프레임 처리와 비동기 마스크 그리기 연결
    /** 그리기 결과가 확정된 기록만 전송 대기열에 넣는다. */
    function track(row, drawing = Promise.resolve({ status: "skipped", drawn_ms: null })) {
      const task = drawing.then((result) => {
        row.overlay_status = result.status;
        if (result.status === "error") { lastDrawn = null; lastCapture = null; }
        if (result.drawn_ms != null) {
          row.overlay_drawn_ms = result.drawn_ms;
          row.response_to_overlay_ms = result.drawn_ms - row.response_received_ms;
          row.capture_to_overlay_ms = result.drawn_ms - row.capture_started_ms;
          row.overlay_interval_ms = lastDrawn == null ? null : result.drawn_ms - lastDrawn;
          row.previous_overlay_age_ms = lastCapture == null ? null : result.drawn_ms - lastCapture;
          lastDrawn = result.drawn_ms;
          lastCapture = row.capture_started_ms;
        }
        // 장시간 오프라인일 때 메모리가 무한히 증가하지 않도록 제한한다.
        if (queue.length >= 500) {
          queue.shift();
          dropped += 1;
          onWarning(`지연 로그 ${dropped}건이 버퍼 초과로 누락되었습니다.`);
        }
        queue.push(row);
      }).finally(() => pending.delete(task));
      pending.add(task);
    }

    // 세션 종료 시 잔여 측정값 저장
    /** 그리기 취소 및 전송 루프 종료 후 호출하며 저장 실패 시 재시도를 유지한다. */
    async function finish() {
      closing = true;
      await Promise.all([...pending]);
      while (batch || queue.length || sending) {
        if (!await flush()) return false;
      }
      clearInterval(timer);
      return true;
    }

    return { track, flush, finish };
  }

  return {
    ApiError, createTimings,
    health: () => request("/health", {}, 4000),
    models: () => request("/models"),
    createSession: (body) => request("/sessions", json(body), 15000),
    getSession: (id) => request(`/sessions/${encodeURIComponent(id)}`),
    retryExport: (id) => request(`/sessions/${encodeURIComponent(id)}/export`, { method: "POST" }),
    resultVideoUrl: (id) => `${BASE}/sessions/${encodeURIComponent(id)}/result-video`,
    stopSession: (id) => request(`/sessions/${encodeURIComponent(id)}/stop`, { method: "POST" }, 10000),
    uploadFrame(id, blob, frameId, capturedAtMs, timeoutMs) {
      const fd = new FormData();
      fd.append("image", blob, `${frameId}.jpg`);
      fd.append("frame_id", String(frameId));
      fd.append("captured_at_ms", String(capturedAtMs));
      fd.append("client_sent_at_ms", String(Date.now()));
      return request(`/sessions/${encodeURIComponent(id)}/frames`, { method: "POST", body: fd }, timeoutMs);
    },
    // 실시간 탐지 영상 업로드
    /** 완성된 WebM 녹화 파일을 실행 중인 세션에 저장한다. */
    uploadRecording(id, blob) {
      const fd = new FormData();
      fd.append("video", blob, "realtime_overlay.webm");
      return request(`/sessions/${encodeURIComponent(id)}/recording`, { method: "POST", body: fd }, 120000);
    },
  };
})();
