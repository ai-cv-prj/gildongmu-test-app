/** 고정된 한국어 안내 음원을 요청 순서대로 끝까지 재생한다. */
(() => {
  const CLIPS = new Map([
    ["신호 안내를 시작합니다.", "startup"],
    ["빨간불입니다.", "red"],
    ["초록불입니다. 다음 초록 신호를 기다려 주세요.", "green-initial-wait"],
    ["초록불입니다.", "green"],
    ["초록불로 바뀌었습니다.", "green-changed"],
    ["빨간불로 바뀌었습니다.", "red-changed"],
    ["신호를 확인할 수 없습니다.", "missing"],
  ]);
  for (const [text, name] of [...CLIPS]) CLIPS.set(`모의 신호. ${text}`, `mock-${name}`);
  CLIPS.set("음성 확인입니다. 이 문장이 들리면 신호 안내를 사용할 수 있습니다.", "sound-check");
  const source = name => `/static/audio/ko-v1/${name}.mp3`;

  function create({ onError = () => {}, onStatus = () => {}, now = () => performance.now() } = {}) {
    // 클릭으로 시작한 재생기를 이후 신호 안내에도 재사용한다.
    const audio = window.Audio ? new window.Audio() : null;
    let current = null, timer = null;
    const queue = [];
    if (audio) {
      audio.preload = "auto";
      audio.volume = 1;
      audio.muted = false;
      audio.src = source("startup");
    }

    function cancel() {
      queue.length = 0;
      const hadCurrent = current !== null;
      current = null;
      clearTimeout(timer);
      timer = null;
      if (audio && hadCurrent) {
        audio.onplaying = audio.onended = audio.onerror = null;
        audio.pause();
        // 로딩 중 취소도 실제 요청과 보류 중인 play()까지 중단한다.
        audio.removeAttribute("src");
        audio.load();
        onStatus("음성 재생을 취소했습니다.");
      }
    }

    function speak(text, validUntil = now() + 8000, { onEnd = () => {} } = {}) {
      if (!audio || !CLIPS.has(text)) {
        const message = !audio ? "이 브라우저는 음성 재생을 지원하지 않습니다." : "안내 음원이 없습니다. 페이지를 새로고침해 주세요.";
        onStatus(message);
        onError(message);
        return false;
      }
      if (now() >= validUntil) return false;
      // 유효한 관측으로 요청된 안내는 대기 중 만료시키지 않는다.
      // 음원 로딩 제한 시간은 앞선 안내가 끝난 뒤 재생을 시도할 때부터 센다.
      queue.push({ text, onEnd, startTimeoutMs: validUntil - now(), started: false });
      return current ? true : playNext();
    }

    function playNext() {
      if (current || !queue.length) return true;
      const request = queue.shift();
      const validUntil = now() + request.startTimeoutMs;
      current = request;
      onStatus("음성 재생을 준비하고 있습니다.");
      const fail = message => {
        if (current !== request) return;
        cancel();
        onStatus(message);
        onError(message);
      };
      const started = () => {
        if (current !== request || request.started) return;
        if (now() >= validUntil) {
          fail("음성 재생이 지연되어 안내를 중단했습니다. 연결 상태를 확인하고 다시 시작해 주세요.");
          return;
        }
        request.started = true;
        clearTimeout(timer);
        onStatus("음성 재생 중입니다. 들리지 않으면 미디어 음량과 연결된 이어폰을 확인해 주세요.");
        timer = setTimeout(() => fail("음성 재생이 끝나지 않아 중단했습니다. 다시 시작해 주세요."), 15000);
      };
      const rejected = error => {
        const messages = {
          NotAllowedError: "브라우저가 음성 재생을 차단했습니다. Chrome의 사이트 소리 허용을 확인하고 음성 확인 버튼을 직접 눌러 주세요.",
          NotSupportedError: "안내 음원을 불러오거나 재생할 수 없습니다. 페이지를 새로고침해 주세요.",
        };
        fail(messages[error?.name] || "음성을 재생하지 못했습니다. 연결 상태와 소리 설정을 확인해 주세요.");
      };
      audio.onplaying = started;
      audio.onended = () => {
        if (current !== request || !request.started || !audio.ended) return;
        clearTimeout(timer);
        timer = null;
        current = null;
        audio.onplaying = audio.onended = audio.onerror = null;
        onStatus("음성 재생이 끝났습니다.");
        request.onEnd();
        playNext();
      };
      audio.onerror = () => {
        if (!audio.error) return;
        fail(audio.error.code === 2
          ? "안내 음원을 불러오지 못했습니다. 서버 연결 상태를 확인해 주세요."
          : "안내 음원 파일을 재생할 수 없습니다. 페이지를 새로고침해 주세요.");
      };
      timer = setTimeout(() => fail("음성 재생이 지연되어 안내를 중단했습니다. 연결 상태를 확인하고 다시 시작해 주세요."),
        Math.max(0, validUntil - now()));
      try {
        audio.src = source(CLIPS.get(request.text));
        audio.load();
        // await 없이 클릭 처리 중 호출해야 모바일의 사용자 동작으로 인정된다.
        const playing = audio.play();
        playing?.then(started, rejected);
        return true;
      } catch (error) {
        rejected(error);
        return false;
      }
    }

    return { speak, cancel };
  }
  window.GTts = { create };
})();
