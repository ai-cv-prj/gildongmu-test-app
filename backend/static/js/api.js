// 서버 API 호출 래퍼
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
    try { body = await res.json(); } catch (_) { /* no body */ }
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

  return {
    ApiError,
    health: () => request("/health", {}, 4000),
    models: () => request("/models"),
    createSession: (body) => request("/sessions", json(body), 15000),
    stopSession: (id) => request(`/sessions/${encodeURIComponent(id)}/stop`, { method: "POST" }, 10000),
    uploadFrame(id, blob, frameId, capturedAtMs, timeoutMs) {
      const fd = new FormData();
      fd.append("image", blob, `${frameId}.jpg`);
      fd.append("frame_id", String(frameId));
      fd.append("captured_at_ms", String(capturedAtMs));
      fd.append("client_sent_at_ms", String(Date.now()));
      return request(`/sessions/${encodeURIComponent(id)}/frames`, { method: "POST", body: fd }, timeoutMs);
    },
  };
})();
