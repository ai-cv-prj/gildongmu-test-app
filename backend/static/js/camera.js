// 후면 카메라 제어와 프레임 캡처
window.GCamera = (() => {
  const video = document.getElementById("video");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: false });
  let stream = null;
  let onEnded = null;

  async function start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("이 브라우저는 카메라를 지원하지 않습니다. HTTPS 주소로 접속했는지 확인하세요.");
    }
    stop();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (e) {
      if (e.name === "NotAllowedError") throw new Error("카메라 권한이 거부되었습니다. 브라우저 설정에서 허용해주세요.");
      if (e.name === "NotFoundError" || e.name === "OverconstrainedError") throw new Error("후면 카메라를 찾지 못했습니다.");
      throw new Error(`카메라를 열 수 없습니다: ${e.message || e.name}`);
    }
    const track = stream.getVideoTracks()[0];
    track.addEventListener("ended", () => { if (onEnded) onEnded(); });
    video.srcObject = stream;
    await new Promise((resolve) => {
      if (video.readyState >= 2) return resolve();
      video.onloadedmetadata = () => resolve();
    });
    try { await video.play(); } catch (_) { /* autoplay 정책 */ }
    return { width: video.videoWidth, height: video.videoHeight, label: track.label };
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
  }

  function active() {
    return !!stream && stream.getVideoTracks().some((t) => t.readyState === "live") && video.videoWidth > 0;
  }

  // 현재 프레임을 긴 변 maxSide 로 축소한 JPEG Blob 으로 만든다
  function capture(maxSide, quality) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return Promise.resolve(null);
    const scale = Math.min(1, maxSide / Math.max(vw, vh));
    const w = Math.round(vw * scale), h = Math.round(vh * scale);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.drawImage(video, 0, 0, w, h);
    return new Promise((resolve) => canvas.toBlob((b) => resolve(b), "image/jpeg", quality));
  }

  return { start, stop, active, capture, video, setOnEnded: (fn) => { onEnded = fn; } };
})();
