/**
 * file_path: backend/static/js/recorder.js
 * 휴대폰 카메라와 현재 탐지 오버레이를 합쳐 WebM 영상으로 녹화한다.
 */
window.GRecorder = (() => {
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  let recorder = null;
  let chunks = [];
  let animationId = 0;

  // 현재 브라우저가 지원하는 WebM 형식 선택
  /** MediaRecorder에서 사용할 수 있는 WebM MIME 형식을 반환한다. */
  function supportedMimeType() {
    return ["video/webm;codecs=vp8", "video/webm"]
      .find((type) => MediaRecorder.isTypeSupported(type)) || "";
  }

  // 카메라와 현재 탐지 화면 합성
  /** 카메라 위에 현재 오버레이를 그려 녹화 프레임을 만든다. */
  function drawFrame() {
    if (!recorder || recorder.state === "inactive") return;
    const width = canvas.width;
    const height = canvas.height;
    ctx.fillStyle = "#121722";
    ctx.fillRect(0, 0, width, height);

    const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
    const videoWidth = video.videoWidth * scale;
    const videoHeight = video.videoHeight * scale;
    ctx.drawImage(video, (width - videoWidth) / 2, (height - videoHeight) / 2, videoWidth, videoHeight);
    ctx.drawImage(overlay, 0, 0, overlay.width, overlay.height, 0, 0, width, height);
    animationId = requestAnimationFrame(drawFrame);
  }

  // 실시간 탐지 화면 녹화 시작
  /** 최대 긴 변 720px, 약 30FPS로 합성 화면 녹화를 시작한다. */
  function start() {
    if (!window.MediaRecorder || !canvas.captureStream) {
      throw new Error("이 브라우저는 화면 녹화를 지원하지 않습니다.");
    }
    const displayWidth = overlay.clientWidth || video.videoWidth;
    const displayHeight = overlay.clientHeight || video.videoHeight;
    const scale = Math.min(1, 720 / Math.max(displayWidth, displayHeight));
    canvas.width = Math.max(2, Math.round(displayWidth * scale / 2) * 2);
    canvas.height = Math.max(2, Math.round(displayHeight * scale / 2) * 2);

    chunks = [];
    const mimeType = supportedMimeType();
    recorder = new MediaRecorder(canvas.captureStream(30), {
      ...(mimeType ? { mimeType } : {}),
      videoBitsPerSecond: 2500000,
    });
    recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    recorder.start(1000);
    drawFrame();
  }

  // 녹화 종료 및 영상 생성
  /** 녹화를 끝내고 서버로 보낼 WebM Blob을 반환한다. */
  function stop() {
    if (!recorder || recorder.state === "inactive") return Promise.resolve(null);
    return new Promise((resolve) => {
      const current = recorder;
      current.onstop = () => {
        cancelAnimationFrame(animationId);
        const blob = new Blob(chunks, { type: current.mimeType || "video/webm" });
        recorder = null;
        chunks = [];
        resolve(blob.size ? blob : null);
      };
      current.stop();
    });
  }

  return { start, stop };
})();
