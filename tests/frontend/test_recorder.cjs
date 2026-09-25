/**
 * file_path: tests/frontend/test_recorder.cjs
 * 녹화용 합성의 30FPS 제한과 종료·재시작 동작을 검사한다.
 * 가짜 화면 갱신 시계를 사용하며 실제 카메라나 파일은 만들지 않는다.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");

// 녹화기 테스트 환경 생성
/** 화면 갱신 시각, 합성 횟수와 녹화 설정을 제어하는 환경을 반환한다. */
function harness() {
  let now = 0, nextId = 0;
  const callbacks = new Map(), draws = [], captureRates = [], recorders = [];
  const video = { videoWidth: 1280, videoHeight: 720 };
  const overlay = { width: 824, height: 892, clientWidth: 412, clientHeight: 446 };
  const canvas = {
    getContext: () => ({ fillRect() {}, drawImage: (source) => draws.push({ time: now, source }) }),
    captureStream: (fps) => { captureRates.push(fps); return {}; },
  };

  class FakeMediaRecorder {
    // 지원 형식 응답
    /** 기존 VP8 WebM 선택 경로를 사용한다. */
    static isTypeSupported() { return true; }

    // 녹화 설정 보관
    /** 실제 인코딩 대신 설정과 상태만 기록한다. */
    constructor(stream, options) {
      this.options = options;
      this.mimeType = options.mimeType;
      this.state = "inactive";
      recorders.push(this);
    }

    // 가짜 녹화 시작
    /** MediaRecorder의 녹화 상태 전환을 재현한다. */
    start() { this.state = "recording"; }

    // 가짜 녹화 종료
    /** 영상 데이터와 종료 이벤트를 전달한다. */
    stop() {
      this.state = "inactive";
      this.ondataavailable({ data: new Blob(["test-video"]) });
      this.onstop();
    }
  }

  const context = {
    window: { MediaRecorder: FakeMediaRecorder }, MediaRecorder: FakeMediaRecorder, Blob,
    document: { getElementById: (id) => id === "video" ? video : overlay, createElement: () => canvas },
    performance: { now: () => now },
    requestAnimationFrame: (fn) => { callbacks.set(++nextId, fn); return nextId; },
    cancelAnimationFrame: (id) => callbacks.delete(id),
  };
  vm.runInNewContext(fs.readFileSync("backend/static/js/recorder.js", "utf8"), context);

  // 화면 갱신 이벤트 실행
  /** 예약된 콜백만 한 차례 실행하여 실제 rAF 동작을 재현한다. */
  function tick(time) {
    now = time;
    const scheduled = [...callbacks.values()];
    callbacks.clear();
    for (const callback of scheduled) callback(time);
  }

  return { recorder: context.window.GRecorder, tick, draws, callbacks, captureRates, recorders, video, overlay, canvas };
}

for (const hz of [30, 60, 120, 144]) {
  test(`${hz}Hz 화면에서도 녹화 합성은 초당 약 30회로 제한된다`, async () => {
    const h = harness();
    h.recorder.start();
    for (let frame = 1; frame < hz * 10; frame++) h.tick(frame * 1000 / hz);
    const composed = h.draws.length / 2;
    assert.ok(composed >= 298 && composed <= 300, `10초 동안 합성 ${composed}회`);
    assert.equal(h.callbacks.size, 1);
    // 합성마다 현재 영상과 마스크가 모두 포함되어야 한다.
    for (let index = 0; index < h.draws.length; index += 2) {
      assert.equal(h.draws[index].source, h.video);
      assert.equal(h.draws[index + 1].source, h.overlay);
    }
    await h.recorder.stop();
  });
}

test("긴 중단 뒤에도 밀린 합성을 몰아서 실행하지 않는다", async () => {
  const h = harness();
  h.recorder.start();
  h.tick(5);
  assert.equal(h.draws.length, 2);
  h.tick(5000);
  assert.equal(h.draws.length, 4);
  assert.equal(h.callbacks.size, 1);
  await h.recorder.stop();
});

test("녹화 설정을 유지하고 종료 시 WebM과 타이머를 정리한다", async () => {
  const h = harness();
  h.recorder.start();
  assert.equal(h.recorder.active(), true);
  assert.deepEqual(h.captureRates, [30]);
  assert.equal(h.recorders[0].options.videoBitsPerSecond, 2500000);
  assert.ok(Math.max(h.canvas.width, h.canvas.height) <= 720);
  h.tick(40);
  const count = h.draws.length;
  const video = await h.recorder.stop();
  assert.equal(video.type, "video/webm;codecs=vp8");
  assert.equal(await video.text(), "test-video");
  assert.equal(h.recorder.active(), false);
  assert.equal(h.callbacks.size, 0);
  h.tick(1000);
  assert.equal(h.draws.length, count);
  assert.equal(await h.recorder.stop(), null);
});

test("새 세션은 이전 프레임 주기를 기다리지 않고 즉시 합성한다", async () => {
  const h = harness();
  h.recorder.start();
  h.tick(40);
  await h.recorder.stop();
  h.tick(41);
  const count = h.draws.length;
  h.recorder.start();
  assert.equal(h.draws.length, count + 2);
  h.tick(42);
  assert.equal(h.draws.length, count + 2);
  assert.equal(h.callbacks.size, 1);
  await h.recorder.stop();
});
