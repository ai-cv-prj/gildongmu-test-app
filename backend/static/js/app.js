/**
 * 파일 경로: backend/static/js/app.js
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
    guidancePanel: $("guidance-panel"), guidanceText: $("guidance-text"),
    ttsStatus: $("tts-status"), btnSoundCheck: $("btn-sound-check"),
    btnMute: $("btn-mute"),
    summary: $("summary"), sId: $("s-id"), sFrames: $("s-frames"), sAvg: $("s-avg"), sPath: $("s-path"),
    alert: $("alert"), exportStatus: $("export-status"), maskToggle: $("walking-mask-toggle"),
  };

  // 기능별로 검증한 입력 크기와 전송 상한을 유지한다. 요청은 항상 하나씩 보낸다.
  const SETTINGS = { confidence: 0.4, image_max_side: 640, target_fps: 10, jpeg_quality: 0.8, timeout_ms: 5000, retry_wait_ms: 500 };
  const TRAFFIC_SETTINGS = { confidence: 0.25, image_max_side: 960, target_fps: 5 };
  // 위험 판단 프로파일과 같은 검출 기준. 입력 크기·전송 상한은 공통값을 쓴다.
  const WALKING_SETTINGS = { confidence: 0.25 };
  const MODE_LABEL = { traffic: "신호등", walking: "도보 장애물", bus: "버스" };

  const state = {
    mode: "traffic", models: [], cameraOn: false, sessionId: null, running: false,
    frameId: 0, sent: 0, failed: 0, recvTimes: [], starting: false, stopping: false,
    timings: null, loopTask: null, logWarning: "",
    voiceEnabled: true,
  };

  const player = GTts.create({
    onError: (message) => guidance.stop(message),
    onStatus: (message) => { el.ttsStatus.textContent = message; },
  });
  const guidance = GGuidance.create({
    player,
    onChange: ({ text }) => {
      el.guidanceText.textContent = text;
      refreshButtons();
    },
  });
  setInterval(() => guidance.tick(), 200);

  // ---------- 상태 표시 ----------
  function setStatus(node, s, text) { node.dataset.state = s; node.lastChild.textContent = text; }
  function setLabel(btn, text) { btn.querySelector(".btn-label").textContent = text; }
  function setBadge(s, text) { el.badge.dataset.state = s; el.badge.textContent = text; }
  function showAlert(msg, tone = "error") {
    if (!msg && state.logWarning) { msg = state.logWarning; tone = "info"; }
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
    const dets = (res.detections || []).filter(d => d.class_name !== "crosswalk");
    el.resultRow.innerHTML = dets.length
      ? dets.map((d) => {
        const display = GOverlay.describeDetection(d);
        return `<span class="chip">${display.label}${display.confidenceText ? `<b>${display.confidenceText}</b>` : ""}</span>`;
      }).join("")
      : `<span class="muted">검출 없음 · frame ${res.frame_id}</span>`;
    const ev = res.event || {};
    if (ev.risk_schema_version) {
      const c = ev.counts || {};
      el.resultRow.textContent = `위험 ${c.danger || 0} · 주의 ${(c.caution || 0) + (c.surface || 0) + (c.advisories || 0)} · 관측 ${c.monitor || 0} · 추적 ${c.tracked || 0}`;
    }
    if (ev.type === "walking_warning" && ev.camera_view?.status === "unavailable") {
      el.resultRow.textContent = "촬영 불가 · 검출 결과 보류 · frame " + res.frame_id;
    } else if (ev.type === "walking_warning" && Number.isFinite(ev.walkable_ratio)) {
      const crosswalk = Number.isFinite(ev.crosswalk_ratio)
        ? ` · 횡단보도(핑크) ${(ev.crosswalk_ratio * 100).toFixed(1)}%` : "";
      el.resultRow.textContent = `보행가능(초록) ${(ev.walkable_ratio * 100).toFixed(1)}%${crosswalk} · frame ${res.frame_id}`;
    }
    let text = "", tone = "";
    if (ev.type === "traffic_signal") {
      tone = ev.signal_state === "green" ? "green" : ev.signal_state === "red" ? "red" : "";
      text = { green: "초록불", red: "빨간불", unknown: "신호 인식 안 됨" }[ev.signal_state] || "";
      // 새 응답에서는 검출 성공과 안내 대상 선택 성공을 구분한다.
      if (Object.prototype.hasOwnProperty.call(ev, "selected_detection_index")) {
        const count = ev.detected_signal_count ?? dets.filter(d => d.class_name === "pedestrian_signal").length;
        let detail;
        if (!count) detail = "신호등 검출 없음";
        else if (ev.selected_detection_index != null) {
          detail = ev.signal_state === "unknown" ? "안내 대상 색상 미확인" : `안내 대상 ${text}`;
        } else {
          detail = ev.candidate_detection_index != null ? "안내 대상 선택 확인 중" : "안내 대상 선택 불가";
        }
        text = count ? `신호등 ${count}개 검출 · ${detail}` : detail;
      }
    } else if (ev.type === "walking_warning") {
      if (ev.warning) { tone = ev.level === "danger" ? "red" : "warn"; text = ev.warning_text || "장애물 주의"; }
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
      if (typeof p.voiceEnabled === "boolean") state.voiceEnabled = p.voiceEnabled;
      if (p.mode && MODE_LABEL[p.mode]) selectMode(p.mode, false);
      if (p.device) {
        const has = [...el.deviceSelect.options].some((o) => o.value === p.device);
        if (has) el.deviceSelect.value = p.device;
        else { el.deviceSelect.value = "__custom__"; el.deviceCustom.value = p.device; }
      }
    } catch (_) { /* 설정 읽기 실패는 무시 */ }
    el.deviceCustom.hidden = el.deviceSelect.value !== "__custom__";
    if (!state.running && !state.starting) el.guidanceText.textContent = idleVoiceText();
    refreshButtons();
  }
  function savePrefs() {
    try { localStorage.setItem("gildongmu.prefs", JSON.stringify({ mode: state.mode, device: currentDevice(), voiceEnabled: state.voiceEnabled })); } catch (_) { /* 설정 저장 실패는 무시 */ }
  }
  function idleVoiceText() {
    const action = state.mode === "walking" ? "위험 장애물을 안내합니다" : "신호를 읽습니다";
    return state.voiceEnabled ? `음성 안내가 켜져 있습니다. 테스트를 시작하면 ${action}.` : "음성 안내가 꺼져 있습니다.";
  }

  function selectMode(mode, refill = true) {
    state.mode = mode;
    el.maskToggle.hidden = mode !== "walking";
    [...el.modeSeg.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    if (refill) fillModels();
  }

  function fillModels() {
    const list = state.models.filter((m) => m.mode === state.mode);
    el.modelSelect.innerHTML = list.map((m) =>
      `<option value="${m.id}" ${m.available ? "" : "disabled"}>${m.name}${m.available ? "" : " (가중치 없음)"}</option>`).join("");
    const preferred = state.mode === "traffic" ? "traffic-best-yolo-v2" : null;
    const first = list.find((m) => m.available && m.id === preferred)
      || list.find((m) => m.available && !m.is_mock) || list.find((m) => m.available);
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
    el.btnStart.disabled = !ready || state.running || state.starting;
    el.btnStart.hidden = state.running;
    el.btnStop.hidden = !state.running;
    // 테스트 중에는 숨기되, 카메라가 끊기면 다시 켤 수 있게 보여준다
    el.btnCamera.hidden = state.running && state.cameraOn;
    // 테스트 중에는 기능·모델을 바꿀 수 없다: 설정을 접고 한 줄 요약만 보여준다
    const locked = state.running || !!state.sessionId;
    el.setup.hidden = locked;
    el.runInfo.hidden = !locked;
    el.guidancePanel.hidden = !["traffic","walking"].includes(state.mode);
    el.btnMute.disabled = !["traffic","walking"].includes(state.mode);
    el.btnMute.textContent = state.voiceEnabled ? "음성 끄기" : "음성 켜기";
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
    guidance.stop(state.running && state.voiceEnabled
      ? "카메라 재시작 후 음성을 껐다 켜서 안내를 재개해 주세요." : idleVoiceText());
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
    guidance.stop("카메라가 끊겨 음성 안내를 중단했습니다. 카메라를 다시 켠 뒤 음성을 껐다 켜 주세요.");
    state.cameraOn = false;
    el.frame.classList.remove("live");
    setBadge("paused", "카메라 끊김");
    if (state.running) showAlert("카메라가 종료되었습니다. 카메라를 다시 시작하면 전송이 재개됩니다.", "info");
    refreshButtons();
  });

  // ---------- 세션 ----------
  async function startSession() {
    if (state.starting || state.running || state.sessionId) return;
    state.starting = true;
    showAlert(null);
    el.summary.hidden = true;
    el.btnStart.disabled = true;
    const settings = { ...SETTINGS, ...(state.mode === "traffic" ? TRAFFIC_SETTINGS
      : state.mode === "walking" ? WALKING_SETTINGS : {}) };
    const body = {
      mode: state.mode,
      model_id: el.modelSelect.value,
      device_type: currentDevice(),
      note: el.note.value.trim(),
      settings: { confidence: settings.confidence, image_max_side: settings.image_max_side,
        target_fps: settings.target_fps, jpeg_quality: settings.jpeg_quality },
      client: { user_agent: navigator.userAgent, screen_width: screen.width, screen_height: screen.height, platform: navigator.platform || "", app_version: "walking-risk-v1" },
    };
    try {
      // 켜기로 선택된 경우 사용자 클릭 안에서 재생을 요청한다.
      const selectedModel = state.models.find(m => m.id === body.model_id);
      if (["traffic","walking"].includes(state.mode) && state.voiceEnabled) {
        guidance.start(null, !!selectedModel?.is_mock, state.mode);
      }
      else guidance.stop();
      const r = await GApi.createSession(body);
      state.sessionId = r.session_id;
      guidance.bindSession(r.session_id);
      state.logWarning = "";
      state.timings = GApi.createTimings(r.session_id, (message) => {
        if (state.sessionId !== r.session_id) return;
        state.logWarning = message;
        if (message) showAlert(message, "info");
      });
      state.running = true;
      state.frameId = 0; state.sent = 0; state.failed = 0; state.recvTimes = [];
      updateMetrics(null, null);
      savePrefs();
      const model = state.models.find((x) => x.id === body.model_id);
      el.runInfo.innerHTML = `<span>${MODE_LABEL[state.mode]}</span><span>${model ? model.name : body.model_id}</span><span>${body.device_type}</span>`;
      setBadge("running", "테스트 중");
      try {
        GRecorder.start();
      } catch (recordError) {
        showAlert(`실시간 녹화를 시작하지 못했습니다: ${recordError.message}`, "info");
      }
      refreshButtons();
      window.scrollTo({ top: 0, behavior: "smooth" });
      state.loopTask = loop(r.session_id, state.timings, settings);
    } catch (e) {
      guidance.stop("테스트를 시작하지 못해 음성 안내를 종료했습니다.");
      if (e.code === "session_conflict" && e.detail && e.detail.active_session) {
        showAlert(`이미 실행 중인 세션이 있습니다 (${e.detail.active_session.id}). 서버에서 종료 후 다시 시도하세요.`);
      } else {
        showAlert(`테스트를 시작하지 못했습니다: ${e.message}`);
      }
      refreshButtons();
    } finally {
      state.starting = false;
      refreshButtons();
    }
  }

  async function stopSession() {
    if (!state.sessionId) return;
    guidance.stop(idleVoiceText());
    state.running = false;
    state.stopping = true;
    refreshButtons();
    el.btnStop.hidden = false;
    el.btnStart.hidden = true;
    el.btnStop.disabled = true;
    setLabel(el.btnStop, "종료 중…");
    setBadge("camera", "종료 중");
    try {
      const recording = await GRecorder.stop();
      if (recording) {
        try {
          await GApi.uploadRecording(state.sessionId, recording);
        } catch (recordError) {
          showAlert(`실시간 녹화 저장 실패: ${recordError.message}`);
        }
      }
      // 마지막 요청과 PNG 디코딩을 정리한 뒤 마지막 묶음까지 저장한다.
      await state.loopTask;
      GOverlay.clear();
      if (!await state.timings.finish()) throw new Error("지연 로그가 아직 저장되지 않았습니다");
      const r = await GApi.stopSession(state.sessionId);
      const s = r.session;
      el.sId.textContent = s.id;
      el.sFrames.textContent = s.mode === "walking" && s.export
        ? `원본 ${s.raw_frame_count}장 · 분석 성공 ${s.frame_count} · 실패 ${s.error_count}`
        : `${s.frame_count}장 저장 · 실패 ${s.error_count}`;
      document.querySelector(".summary-title").lastChild.textContent = s.export ? "원본 저장 · 결과 영상 생성 중" : "저장 완료";
      el.exportStatus.hidden = !s.export;
      if (s.export) watchExport(s.id);
      el.sAvg.textContent = `${fmt(s.average_inference_ms)} ms / ${fmt(s.average_server_ms)} ms (p95 ${fmt(s.p95_server_ms)} ms)`;
      el.sPath.textContent = s.storage_path;
      el.summary.hidden = false;
      state.sessionId = null;
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

  /** 한 번에 한 프레임을 보내며 각 단계의 브라우저 시간을 기록한다. */
  async function loop(sessionId, timings, settings) {
    const interval = 1000 / settings.target_fps;
    let paused = false;
    let previousCapture = null;
    while (state.running) {
      if (document.hidden || !GCamera.active()) {
        if (!paused) { paused = true; setBadge("paused", document.hidden ? "백그라운드 · 일시정지" : "카메라 대기"); }
        await sleep(300);
        continue;
      }
      if (paused) { paused = false; setBadge("running", "테스트 중"); showAlert(null); }

      const t0 = performance.now();
      // 보행 위험 판단은 프레임 간격으로 운동을 계산하므로 벽시계 대신 단조 증가 시각을 쓴다.
      const capturedAt = state.mode === "walking"
        ? Math.round(performance.timeOrigin + performance.now())
        : Date.now(); // JPEG 생성 이전 시각으로 서버 로그와 연결한다.
      const frameId = ++state.frameId;
      const row = {
        frame_id: frameId, captured_at_ms: capturedAt, capture_started_ms: t0,
        capture_ms: 0,
        capture_interval_ms: previousCapture == null ? null : t0 - previousCapture,
        jpeg_bytes: 0, recording_active: GRecorder.active(),
        visibility: document.visibilityState, status: "cancelled", throttle_wait_ms: 0, retry_wait_ms: 0,
      };
      previousCapture = t0;
      let drawing;
      try {
        const blob = await GCamera.capture(settings.image_max_side, settings.jpeg_quality);
        row.capture_ms = performance.now() - t0;
        row.capture_backend = GCamera.captureBackend?.() || "canvas";
        row.jpeg_bytes = blob?.size || 0;
        if (!state.running) break;
        if (!blob) {
          guidance.interrupt();
          row.status = "error"; row.error_code = "capture_empty";
          const waitStarted = performance.now();
          await sleep(100);
          row.retry_wait_ms = performance.now() - waitStarted;
          continue;
        }
        row.request_started_ms = performance.now();
        const res = await GApi.uploadFrame(sessionId, blob, frameId, capturedAt, SETTINGS.timeout_ms);
        row.response_received_ms = performance.now();
        row.request_ms = row.response_received_ms - row.request_started_ms;
        row.http_status = 200;
        if (!state.running) break;
        row.status = "ok";
        state.sent += 1;
        state.recvTimes.push(row.response_received_ms);
        if (res.frame_id === frameId) {
          drawing = GOverlay.draw(res.detections, res.event);
          showResult(res);
          guidance.accept(res, t0);
        }
        // 화면의 기존 '왕복' 표시는 캡처를 포함한 값으로 유지한다.
        updateMetrics(res.timing, row.response_received_ms - t0);
        if (!res.saved) setStatus(el.storageStatus, "warn", "저장 꺼짐"); else setStatus(el.storageStatus, "ok", "저장");
        setStatus(el.serverStatus, "ok", "서버");
        showAlert(null);
        const remain = interval - (performance.now() - t0);
        if (remain > 0) {
          const waitStarted = performance.now();
          await sleep(remain);
          row.throttle_wait_ms = performance.now() - waitStarted;
        }
      } catch (e) {
        guidance.interrupt();
        row.status = "error";
        row.error_code = e.code || (row.request_started_ms == null ? "capture_failed" : "client_error");
        row.http_status = e.status || 0;
        if (row.request_started_ms == null) row.capture_ms = performance.now() - t0;
        if (row.request_ms == null && row.request_started_ms != null) {
          row.request_ms = performance.now() - row.request_started_ms;
        }
        if (!state.running) break;
        state.failed += 1;
        updateMetrics(null, null);
        if (e.status === 409 || e.status === 404) {
          showAlert(`세션이 서버에서 종료되었습니다 (${e.message}). 테스트를 다시 시작하세요.`);
          state.running = false; state.sessionId = null;
          guidance.stop();
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
          showAlert(`프레임 처리 오류: ${e.message}`);
        }
        const waitStarted = performance.now();
        await sleep(SETTINGS.retry_wait_ms);
        row.retry_wait_ms = performance.now() - waitStarted;
      } finally {
        timings.track(row, drawing);
      }
    }
    if (!state.stopping) { GOverlay.clear(); void timings.finish(); }
  }

  async function watchExport(id) {
    if (el.sId.textContent !== id) return;
    try {
      const s = await GApi.getSession(id);
      if (el.sId.textContent !== id) return;
      const e = s.export || {};
      el.exportStatus.replaceChildren();
      const title = document.querySelector(".summary-title").lastChild;
      if (e.state === "ready") {
        title.textContent = "원본·결과 영상 저장 완료";
        const link = document.createElement("a");
        link.href = GApi.resultVideoUrl(id);
        link.textContent = `결과 영상 다운로드 · 원본 ${e.frame_count}장`;
        el.exportStatus.append(link);
      } else if (e.state === "failed") {
        title.textContent = "원본 저장 · 결과 영상 생성 실패";
        el.exportStatus.append(document.createTextNode(e.error || "결과 영상 생성 실패"));
        const retry = document.createElement("button");
        retry.type = "button"; retry.textContent = "영상 생성 다시 시도";
        retry.onclick = async () => {
          retry.disabled = true;
          try { await GApi.retryExport(id); watchExport(id); }
          catch (err) { retry.disabled = false; showAlert(err.message); }
        };
        el.exportStatus.append(retry);
      } else {
        el.exportStatus.textContent = "결과 영상 생성 중…";
        setTimeout(() => watchExport(id), 1500);
      }
    } catch (e) {
      el.exportStatus.textContent = "결과 영상 상태 확인 중…";
      setTimeout(() => watchExport(id), 3000);
    }
  }
  el.maskToggle.querySelector("input").addEventListener("change", (e) => GOverlay.setWalkingMaskEnabled(e.target.checked));

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
  el.btnSoundCheck.addEventListener("click", () => {
    player.speak("음성 확인입니다. 이 문장이 들리면 신호 안내를 사용할 수 있습니다.");
  });
  el.btnMute.addEventListener("click", () => {
    if (!["traffic","walking"].includes(state.mode)) return;
    state.voiceEnabled = !state.voiceEnabled;
    savePrefs();
    if (state.voiceEnabled && (state.running || state.starting) && !state.stopping && state.cameraOn && !document.hidden) {
      const model = state.models.find(m => m.id === el.modelSelect.value);
      guidance.start(state.sessionId, !!model?.is_mock, state.mode);
    } else guidance.stop(state.voiceEnabled && (state.running || state.starting)
      ? "음성 안내가 켜져 있습니다. 화면과 카메라가 준비되면 음성을 껐다 켜 주세요." : idleVoiceText());
  });
  // 화면 잠금이나 앱 전환에서 돌아왔을 때 카메라가 죽어 있으면 재시작 버튼을 보여준다
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) guidance.stop("화면이 숨겨져 음성 안내를 중단했습니다. 돌아온 뒤 음성을 껐다 켜 주세요.");
    refreshButtons();
    if (document.hidden || !state.cameraOn) return;
    GCamera.video.play().catch(() => {});
    setTimeout(() => {
      if (GCamera.active()) return;
      guidance.stop("카메라가 끊겨 음성 안내를 종료했습니다.");
      state.cameraOn = false;
      el.frame.classList.remove("live");
      setBadge("paused", "카메라 끊김");
      showAlert("카메라가 멈췄습니다. '카메라 재시작'을 누르면 테스트가 이어집니다.", "info");
      refreshButtons();
    }, 800);
  });
  window.addEventListener("beforeunload", (e) => { if (state.sessionId) { e.preventDefault(); e.returnValue = ""; } });
  window.addEventListener("pagehide", () => guidance.stop());

  // ---------- 초기화 ----------
  loadPrefs();
  refreshButtons();
  loadModels().then(() => loadPrefs());
  pollHealth();
  setInterval(pollHealth, 5000);
})();
