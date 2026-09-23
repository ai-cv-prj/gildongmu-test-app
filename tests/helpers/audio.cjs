/** 실제 Promise와 수동 미디어 이벤트로 재생·취소 경합을 재현한다. */
function audioHarness({ onPlay = () => {} } = {}) {
  const instances = [], plays = [], pauses = [];
  class Audio {
    constructor() { instances.push(this); }
    load() { this.ended = false; this.error = null; }
    pause() { pauses.push(this); }
    removeAttribute(name) { if (name === "src") this.src = ""; }
    play() {
      onPlay();
      const onplaying = this.onplaying, onended = this.onended, onerror = this.onerror;
      return new Promise((resolve, reject) => {
        plays.push({
          src: this.src, playbackRate: this.playbackRate, resolve, reject,
          start: () => { onplaying?.(); resolve(); },
          end: () => { this.ended = true; onended?.(); },
          error: (code) => { this.error = { code }; onerror?.(); },
        });
      });
    }
  }
  return { Audio, instances, plays, pauses };
}
module.exports = { audioHarness };
