/**
 * file_path: backend/static/js/app.js
 * 카메라 프레임 전송과 테스트 상태를 관리하고 검출·보행가능·횡단보도 결과를 표시한다.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const el = {
    serverStatus: $("server-status"), storageStatus: $("storage-status"),
    frame: $("camera-frame"), badge: $("session-badge"), banner: $("event-banner"), resultRow: $("result-row"),
    mInference: $("m-inference"), mServer: $("m-server"), mRtt: $("m-rtt"), mFps: $("m-fps"), mFrames: $("m-frames"), mFail: $("m-fail"),
    setup: $("setup-panel"), runInfo: $("run-info"), cameraCard: document.querySelector(".camera-card"), modeSeg: $("mode-seg"), modelSelect: $("model-select"), modelHint: $("model-hint"),
    deviceSelect: $("device-select"), deviceCustom: $("device-custom"), note: $("note"),
    btnCamera: $("btn-camera"), btnStart: $("btn-start"), btnStop: $("btn-stop"),
    summary: $("summary"), sId: $("s-id"), sFrames: $("s-frames"), sAvg: $("s-avg"), sPath: $("s-path"),
    alert: $("alert"),
  };

  const SETTINGS = { confidence: 0.4, image_max_side: 960, target_fps: 5, jpeg_quality: 0.8, timeout_ms: 5000, retry_wait_ms: 500 };
  const MODE_LABEL = { traffic: "신호등", walking: "도보 장애물", bus: "버스" };

  const state = {
    mode: "traffic", models: [], cameraOn: false, sessionId: null, running: false,
    frameId: 0, sent: 0, failed: 0, recvTimes: [], stopping: false,
  };

  // ---------- 상태 표시 ----------
  function setStatus(node, s, text) { node.dataset.state = s; node.lastChild.textContent = text; }
  function setLabel(btn, text) { btn.querySelector(".btn-label").textContent = text; }
  function setBadge(s, text) { el.badge.dataset.state = s; el.badge.textContent = text; }
  function showAlert(msg, tone = "error") {
    if (!msg) { el.alert.hidden = true; return; }
    el.alert.hidden = false; el.alert.dataset.tone = tone; el.alert.textContent = msg;
  }
  function fmt(v) { return v == null ? "–" : Math.round(v).toString(); }

  function updateMetrics(t, rtt) {
    if (t) { el.mInference.textContent = fmt(t.inference_ms); el.mServer.textContent = fmt(t.server_ms); }
    if (rtt != null) el.mRtt.textContent = fmt(rtt);
    el.mFrames.textContent = state.sent;
    el.mFail.textContent = state.failed;
    const now = performance.now();
    state.recvTimes = state.recvTimes.filter((x) => now - x < 3000);
    el.mFps.textContent = state.recvTimes.length ? (state.recvTimes.length / 3).toFixed(1) : "–";
  }

  // 검출 및 보행가능·횡단보도 영역 요약 표시
  /** 결과 종류에 맞춰 검출 목록 또는 보행가능·횡단보도 영역 비율을 보여준다. */
  function showResult(res) {
    const dets = res.detections || [];
    el.resultRow.innerHTML = dets.length
      ? dets.map((d) => {
        const display = GOverlay.describeDetection(d);
        return `<span class="chip">${display.label}${display.confidenceText ? `<b>${display.confidenceText}</b>` : ""}</span>`;
      }).join("")
      : `<span class="muted">검출 없음 · frame ${res.frame_id}</span>`;
    const ev = res.event || {};
    if (ev.type === "walking_warning" && Number.isFinite(ev.walkable_ratio)) {
      const crosswalk = Number.isFinite(ev.crosswalk_ratio)
        ? ` · 횡단보도(핑크) ${(ev.crosswalk_ratio * 100).toFixed(1)}%` : "";
      el.resultRow.textContent = `보행가능(초록) ${(ev.walkable_ratio * 100).toFixed(1)}%${crosswalk} · frame ${res.frame_id}`;
    }
    let text = "", tone = "";
    if (ev.type === "traffic_signal") {
      tone = ev.signal_state === "green" ? "green" : ev.signal_state === "red" ? "red" : "";
      text = { green: "초록불", red: "빨간불", unknown: "신호 인식 안 됨" }[ev.signal_state] || "";
    } else if (ev.type === "walking_warning") {
      if (ev.warning) { tone = "warn"; text = ev.warning_text || "장애물 주의"; }
    } else if (ev.type === "bus_detection") {
      if (ev.bus_number) { tone = ev.is_target ? "green" : ""; text = `버스 ${ev.bus_number}${ev.is_target ? " · 목표 버스" : ""}`; }
    }
    el.banner.hidden = !text; el.banner.dataset.tone = tone; el.banner.textContent = text;
  }

  // ---------- 설정 ----------
  function currentDevice() {
    return el.deviceSelect.value === "__custom__" ? el.deviceCustom.value.trim() : el.deviceSelect.value;
  }
  function loadPrefs() {
    try {
      const p = JSON.parse(localStorage.getItem("gildongmu.prefs") || "{}");
      if (p.mode && MODE_LABEL[p.mode]) selectMode(p.mode, false);
      if (p.device) {
        const has = [...el.deviceSelect.options].some((o) => o.value === p.device);
        if (has) el.deviceSelect.value = p.device;
        else { el.deviceSelect.value = "__custom__"; el.deviceCustom.value = p.device; }
      }
    } catch (_) { /* ignore */ }
    el.deviceCustom.hidden = el.deviceSelect.value !== "__custom__";
  }
  function savePrefs(extra = {}) {
    try { localStorage.setItem("gildongmu.prefs", JSON.stringify({ mode: state.mode, device: currentDevice(), ...extra })); } catch (_) { /* ignore */ }
  }

  function selectMode(mode, refill = true) {
    state.mode = mode;
    [...el.modeSeg.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    if (refill) fillModels();
  }

  function fillModels() {
    const list = state.models.filter((m) => m.mode === state.mode);
    el.modelSelect.innerHTML = list.map((m) =>
      `<option value="${m.id}" ${m.available ? "" : "disabled"}>${m.name}${m.available ? "" : " (가중치 없음)"}</option>`).join("");
    const first = list.find((m) => m.available && !m.is_mock) || list.find((m) => m.available);
    if (first) el.modelSelect.value = first.id;
    updateModelHint();
  }
  function updateModelHint() {
    const m = state.models.find((x) => x.id === el.modelSelect.value);
    el.modelHint.textContent = m ? (m.is_mock ? "가짜 박스로 전체 흐름을 검증합니다" : m.note || "") : "";
    refreshButtons();
  }

  function refreshButtons() {
    const ready = state.cameraOn && !!el.modelSelect.value && currentDevice().length > 0;
    el.btnStart.disabled = !ready || state.running;
    el.btnStart.hidden = state.running;
    el.btnStop.hidden = !state.running;
    // 테스트 중에는 숨기되, 카메라가 끊기면 다시 켤 수 있게 보여준다
    el.btnCamera.hidden = state.running && state.cameraOn;
    // 테스트 중에는 기능·모델을 바꿀 수 없다: 설정을 접고 한 줄 요약만 보여준다
    const locked = state.running || !!state.sessionId;
    el.setup.hidden = locked;
    el.runInfo.hidden = !locked;
  }

  // ---------- 서버 ----------
  async function pollHealth() {
    try {
      const h = await GApi.health();
      setStatus(el.serverStatus, "ok", "서버");
      setStatus(el.storageStatus, h.storage_writable ? "ok" : "bad", h.storage_writable ? "저장" : "저장 불가");
    } catch (_) {
      setStatus(el.serverStatus, "bad", "서버 끊김");
    }
  }

  async function loadModels() {
    try {
      const r = await GApi.models();
      state.models = r.models;
      fillModels();
    } catch (e) {
      showAlert(`모델 목록을 불러오지 못했습니다: ${e.message}`);
    }
  }

  // ---------- 카메라 ----------
  async function startCamera() {
    showAlert(null);
    el.btnCamera.disabled = true;
    try {
      await GCamera.start();
      state.cameraOn = true;
      el.frame.classList.add("live");
      setBadge("camera", "카메라 켜짐");
      setLabel(el.btnCamera, "카메라 재시작");
      el.cameraCard.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      showAlert(e.message);
    } finally {
      el.btnCamera.disabled = false;
      refreshButtons();
    }
  }
  GCamera.setOnEnded(() => {
    state.cameraOn = false;
    el.frame.classList.remove("live");
    setBadge("paused", "카메라 끊김");
    if (state.running) showAlert("카메라가 종료되었습니다. 카메라를 다시 시작하면 전송이 재개됩니다.", "info");
    refreshButtons();
  });

  // ---------- 세션 ----------
  async function startSession() {
    showAlert(null);
    el.summary.hidden = true;
    el.btnStart.disabled = true;
    const body = {
      mode: state.mode,
      model_id: el.modelSelect.value,
      device_type: currentDevice(),
      note: el.note.value.trim(),
      settings: { confidence: SETTINGS.confidence, image_max_side: SETTINGS.image_max_side, target_fps: SETTINGS.target_fps, jpeg_quality: SETTINGS.jpeg_quality },
      client: { user_agent: navigator.userAgent, screen_width: screen.width, screen_height: screen.height, platform: navigator.platform || "" },
    };
    try {
      const r = await GApi.createSession(body);
      state.sessionId = r.session_id;
      state.running = true;
      state.frameId = 0; state.sent = 0; state.failed = 0; state.recvTimes = [];
      updateMetrics(null, null);
      savePrefs();
      const model = state.models.find((x) => x.id === body.model_id);
      el.runInfo.innerHTML = `<span>${MODE_LABEL[state.mode]}</span><span>${model ? model.name : body.model_id}</span><span>${body.device_type}</span>`;
      setBadge("running", "테스트 중");
      refreshButtons();
      window.scrollTo({ top: 0, behavior: "smooth" });
      loop();
    } catch (e) {
      if (e.code === "session_conflict" && e.detail && e.detail.active_session) {
        showAlert(`이미 실행 중인 세션이 있습니다 (${e.detail.active_session.id}). 서버에서 종료 후 다시 시도하세요.`);
      } else {
        showAlert(`테스트를 시작하지 못했습니다: ${e.message}`);
      }
      refreshButtons();
    }
  }

  async function stopSession() {
    if (!state.sessionId) return;
    state.running = false;
    state.stopping = true;
    el.btnStop.disabled = true;
    setLabel(el.btnStop, "종료 중…");
    setBadge("camera", "종료 중");
    try {
      const r = await GApi.stopSession(state.sessionId);
      const s = r.session;
      el.sId.textContent = s.id;
      el.sFrames.textContent = `${s.frame_count}장 저장 · 실패 ${s.error_count}`;
      el.sAvg.textContent = `${fmt(s.average_inference_ms)} ms / ${fmt(s.average_server_ms)} ms (p95 ${fmt(s.p95_server_ms)} ms)`;
      el.sPath.textContent = s.storage_path;
      el.summary.hidden = false;
      state.sessionId = null;
      GOverlay.clear();
      el.banner.hidden = true;
      el.resultRow.innerHTML = '<span class="muted">최근 결과 없음</span>';
      setBadge(state.cameraOn ? "camera" : "idle", state.cameraOn ? "카메라 켜짐" : "대기");
    } catch (e) {
      showAlert(`종료 요청이 실패했습니다: ${e.message}. 다시 '테스트 종료'를 눌러주세요.`);
      state.running = false;
      setBadge("paused", "미종료");
      el.btnStop.hidden = false; el.btnStart.hidden = true;
    } finally {
      state.stopping = false;
      el.btnStop.disabled = false;
      setLabel(el.btnStop, "테스트 종료");
      // 종료 실패 시에는 종료 버튼을 유지한다
      if (!state.sessionId) refreshButtons();
      else { el.btnStop.hidden = false; el.btnStart.hidden = true; }
    }
  }

  // ---------- 전송 루프: 한 번에 요청 하나, 응답 후 최신 프레임 ----------
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function loop() {
    const interval = 1000 / SETTINGS.target_fps;
    let paused = false;
    while (state.running) {
      if (document.hidden || !GCamera.active()) {
        if (!paused) { paused = true; setBadge("paused", document.hidden ? "백그라운드 · 일시정지" : "카메라 대기"); }
        await sleep(300);
        continue;
      }
      if (paused) { paused = false; setBadge("running", "테스트 중"); showAlert(null); }

      const t0 = performance.now();
      const blob = await GCamera.capture(SETTINGS.image_max_side, SETTINGS.jpeg_quality);
      if (!blob) { await sleep(100); continue; }
      const frameId = ++state.frameId;
      const capturedAt = Date.now();
      try {
        const res = await GApi.uploadFrame(state.sessionId, blob, frameId, capturedAt, SETTINGS.timeout_ms);
        if (!state.running) break;
        const rtt = performance.now() - t0;
        state.sent += 1;
        state.recvTimes.push(performance.now());
        if (res.frame_id === frameId) { GOverlay.draw(res.detections); showResult(res); }
        updateMetrics(res.timing, rtt);
        if (!res.saved) setStatus(el.storageStatus, "warn", "저장 꺼짐"); else setStatus(el.storageStatus, "ok", "저장");
        setStatus(el.serverStatus, "ok", "서버");
        showAlert(null);
      } catch (e) {
        if (!state.running) break;
        state.failed += 1;
        updateMetrics(null, null);
        if (e.status === 409 || e.status === 404) {
          showAlert(`세션이 서버에서 종료되었습니다 (${e.message}). 테스트를 다시 시작하세요.`);
          state.running = false; state.sessionId = null;
          setBadge("camera", "카메라 켜짐"); refreshButtons();
          break;
        }
        if (e.code === "storage_failed" || e.status === 507) {
          setStatus(el.storageStatus, "bad", "저장 실패");
          showAlert(`서버 저장 실패: ${e.message}`);
        } else if (e.code === "timeout" || e.code === "network") {
          setStatus(el.serverStatus, "bad", "서버 끊김");
          showAlert(`서버 응답 없음 (${e.message}). 재시도 중…`, "info");
        } else {
          showAlert(`전송 오류: ${e.message}`);
        }
        await sleep(SETTINGS.retry_wait_ms);
        continue;
      }
      const remain = interval - (performance.now() - t0);
      if (remain > 0) await sleep(remain);
    }
  }

  // ---------- 이벤트 ----------
  el.modeSeg.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (b && !state.running) { selectMode(b.dataset.mode); savePrefs(); }
  });
  el.modelSelect.addEventListener("change", updateModelHint);
  el.deviceSelect.addEventListener("change", () => {
    el.deviceCustom.hidden = el.deviceSelect.value !== "__custom__";
    if (!el.deviceCustom.hidden) el.deviceCustom.focus();
    savePrefs(); refreshButtons();
  });
  el.deviceCustom.addEventListener("input", () => { savePrefs(); refreshButtons(); });
  el.btnCamera.addEventListener("click", startCamera);
  el.btnStart.addEventListener("click", startSession);
  el.btnStop.addEventListener("click", stopSession);
  // 화면 잠금이나 앱 전환에서 돌아왔을 때 카메라가 죽어 있으면 재시작 버튼을 보여준다
  document.addEventListener("visibilitychange", () => {
    if (document.hidden || !state.cameraOn) return;
    GCamera.video.play().catch(() => {});
    setTimeout(() => {
      if (GCamera.active()) return;
      state.cameraOn = false;
      el.frame.classList.remove("live");
      setBadge("paused", "카메라 끊김");
      showAlert("카메라가 멈췄습니다. '카메라 재시작'을 누르면 테스트가 이어집니다.", "info");
      refreshButtons();
    }, 800);
  });
  window.addEventListener("beforeunload", (e) => { if (state.running) { e.preventDefault(); e.returnValue = ""; } });

  // ---------- 초기화 ----------
  loadPrefs();
  refreshButtons();
  loadModels().then(() => loadPrefs());
  pollHealth();
  setInterval(pollHealth, 5000);
})();
