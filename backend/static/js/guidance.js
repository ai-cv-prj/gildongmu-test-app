/** 신호 상태와 도보 위험 장애물의 음성 안내 정책. */
(() => {
  const LIMITS = Object.freeze({ stableMs: 400, stableFrames: 3, maxGapMs: 1000, maxAgeMs: 1500,
    missingMs: 2000, walkingRepeatMs: 5000 });

  function create({ player, onChange = () => {}, now = () => performance.now() }) {
    let active = false, sessionId = null, mock = false;
    let initializing = false;
    let startedAt = 0, lastFrame = null, lastCapture = null, lastValid = null;
    let target = null, color = null, candidate = null, missingAnnounced = false;
    let hasConfirmedSignal = false;
    let mode = "traffic", walkingEvent = null, walkingAnnouncedAt = -Infinity;
    // 대상 추적 이력과 분리한다. 소실 안내 후 재확인한 경우에만 같은 색을 다시 읽는다.
    let lastAnnouncedColor = null;

    function update(text) { onChange({ active, text }); }
    function resetEvidence() { target = null; color = null; candidate = null; }
    function announce(text, validUntil) {
      const message = mock && mode === "traffic" ? `모의 신호. ${text}` : text;
      update(message);
      player.speak(message, validUntil);
    }
    function stop(text = "음성 안내가 꺼져 있습니다.") {
      active = false;
      initializing = false;
      hasConfirmedSignal = false;
      lastAnnouncedColor = null;
      resetEvidence();
      player.cancel();
      update(text);
    }
    function start(id, isMock = false, nextMode = "traffic") {
      stop();
      active = true;
      initializing = true;
      mode = nextMode;
      sessionId = id;
      mock = isMock;
      startedAt = now();
      lastFrame = lastCapture = lastValid = null;
      walkingEvent = null;
      walkingAnnouncedAt = -Infinity;
      missingAnnounced = false;
      const message = mode === "walking" ? "장애물 안내를 시작합니다."
        : mock ? "모의 신호. 신호 안내를 시작합니다." : "신호 안내를 시작합니다.";
      update(message);
      // 첫 재생은 사용자 클릭에서 시작하고 완료 전에는 추론 결과로 취소하지 않는다.
      player.speak(message, now() + 8000, { onEnd: () => {
        if (!active || !initializing) return;
        initializing = false;
        startedAt = now();
        update(mode === "walking" ? "위험 장애물을 확인하고 있습니다." : "안내 대상의 신호를 확인하고 있습니다.");
      } });
    }
    function interrupt() {
      if (!active) return;
      if (target !== null || candidate !== null) {
        resetEvidence();
        update("신호를 다시 확인하고 있습니다.");
      }
    }
    // 관측 이력만 초기화한다. 재생 중이거나 대기 중인 안내는 끝까지 이어 읽는다.
    function tick() {
      if (!active || initializing || mode === "walking") return;
      const age = now() - (lastValid ?? startedAt);
      if (age > LIMITS.maxGapMs) interrupt();
      if (hasConfirmedSignal && age >= LIMITS.missingMs && !missingAnnounced) {
        missingAnnounced = true;
        announce("신호를 확인할 수 없습니다.", now() + LIMITS.maxAgeMs);
      }
    }
    function accept(res, capturedAt) {
      if (!active || initializing || res.session_id !== sessionId || capturedAt < startedAt) return;
      const time = now();
      if (!Number.isFinite(capturedAt) || time < capturedAt || time - capturedAt >= LIMITS.maxAgeMs) {
        interrupt();
        return;
      }
      if (!Number.isInteger(res.frame_id) || (lastFrame !== null && res.frame_id <= lastFrame)) return;
      const continuous = lastFrame === null || (res.frame_id === lastFrame + 1 &&
        capturedAt > lastCapture && capturedAt - lastCapture <= LIMITS.maxGapMs);
      if (!continuous) interrupt();
      lastFrame = res.frame_id;
      lastCapture = capturedAt;
      const event = res.event || {};
      if (mode === "walking") {
        if (event.type !== "walking_warning" || event.level !== "danger" ||
            !["person","vehicle","obstacle"].includes(event.voice_category) ||
            !Number.isInteger(event.voice_event_id)) return;
        const key = `${event.voice_event_id}:${event.voice_category}`;
        if (walkingEvent === key && capturedAt-walkingAnnouncedAt < LIMITS.walkingRepeatMs) return;
        walkingEvent = key;
        walkingAnnouncedAt = capturedAt;
        const messages = { person:"위험! 사람이 있음.", vehicle:"위험! 차량이 있음.",
          obstacle:"위험! 장애물이 있음." };
        announce(messages[event.voice_category],capturedAt+LIMITS.maxAgeMs);
        return;
      }
      const index = event.selected_detection_index;
      const selected = Number.isInteger(index) && index >= 0 ? res.detections?.[index] : null;
      if (event.type !== "traffic_signal" || !selected || !Number.isInteger(selected.track_id) ||
          !["red", "green"].includes(event.signal_state)) {
        interrupt();
        return;
      }
      lastValid = capturedAt;
      if (target !== selected.track_id) {
        resetEvidence();
        target = selected.track_id;
        update("안내 대상의 신호를 확인하고 있습니다.");
      }
      const next = event.signal_state;
      if (!candidate || candidate.color !== next) {
        candidate = { color: next, since: capturedAt, count: 1 };
      } else candidate.count++;
      if (candidate.count < LIMITS.stableFrames || capturedAt - candidate.since < LIMITS.stableMs) return;
      if (color === next) return;
      const previous = color;
      color = next;
      const firstConfirmed = !hasConfirmedSignal;
      const recoveredAfterMissing = missingAnnounced;
      // 단순 검출 후보가 아니라 색상을 안정적으로 확인한 뒤부터 소실 안내를 허용한다.
      hasConfirmedSignal = true;
      missingAnnounced = false;
      let text;
      if (previous !== null && previous !== next) {
        text = next === "green" ? "초록불로 바뀌었습니다." : "빨간불로 바뀌었습니다.";
      } else if (next === "green") {
        text = firstConfirmed ? "초록불입니다. 다음 초록 신호를 기다려 주세요."
          : "초록불입니다.";
      } else text = "빨간불입니다.";
      if (lastAnnouncedColor === next && !recoveredAfterMissing) {
        // 소실 안내가 없었던 짧은 끊김·대상 교체에서는 같은 색을 반복하지 않는다.
        update(mock ? `모의 신호. ${text}` : text);
        return;
      }
      lastAnnouncedColor = next;
      announce(text, capturedAt + LIMITS.maxAgeMs);
    }
    // 테스트 시작 클릭에서 음성을 먼저 준비하고 서버 세션 생성 후 결과를 연결한다.
    function bindSession(id) { if (active) sessionId = id; }
    return { start, bindSession, stop, accept, interrupt, tick };
  }
  window.GGuidance = { create, LIMITS };
})();
