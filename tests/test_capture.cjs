/**
 * 파일 경로: tests/test_capture.cjs
 * Worker JPEG 캡처, 미지원 브라우저 전환, 종료 시 자원 해제를 검사한다.
 * 실제 카메라나 폴더를 사용하지 않고 Worker와 canvas를 메모리에서 대체한다.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");

// 브라우저 캡처 환경 생성
/** Worker의 응답·실패·시간 초과를 직접 제어하는 환경을 반환한다. */
function harness({ supported = true, bitmapFactory, postError = false, videoFrameClass } = {}) {
  const workers = [], bitmaps = [], timers = new Map(), fallback = [], warnings = [];
  let timerId = 0;
  const video = { videoWidth: 720, videoHeight: 1280 };
  const canvas = {
    getContext: () => ({ drawImage() {} }),
    toBlob: (callback, type, quality) => {
      fallback.push({ type, quality, width: canvas.width, height: canvas.height });
      callback(new Blob(["fallback"], { type }));
    },
  };
  class FakeWorker {
    // Worker 생성 기록
    /** URL과 전송 메시지를 확인할 수 있게 보관한다. */
    constructor(url) { this.url = url; this.messages = []; workers.push(this); }

    // 프레임 전송 기록
    /** 소유권 이전 목록을 기록하고 전송 실패를 재현할 수 있다. */
    postMessage(data, transfer) {
      if (postError) throw new Error("transfer failed");
      this.messages.push({ data, transfer });
    }

    // Worker 종료 기록
    /** 호출자가 Worker를 정리했는지 검사한다. */
    terminate() { this.terminated = true; }
  }
  // 카메라 프레임 비트맵 생성
  /** 비트맵 자원 해제 횟수를 검사한다. */
  const createBitmap = bitmapFactory || (async () => {
    const bitmap = { closed: 0, close() { this.closed++; } };
    bitmaps.push(bitmap);
    return bitmap;
  });
  const context = {
    window: supported ? { Worker: FakeWorker, OffscreenCanvas: class {}, createImageBitmap: createBitmap, VideoFrame: videoFrameClass } : {},
    Worker: FakeWorker, createImageBitmap: createBitmap, VideoFrame: videoFrameClass,
    performance: { now: () => 12.345 },
    document: { getElementById: () => video, createElement: () => canvas },
    setTimeout: (fn) => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: (id) => timers.delete(id), console: { warn: (...args) => warnings.push(args) },
  };
  vm.runInNewContext(fs.readFileSync("backend/static/js/camera.js", "utf8"), context);
  return { camera: context.window.GCamera, workers, bitmaps, timers, fallback, warnings, video };
}

test("640px·품질 0.8을 Worker에 보내고 JPEG 응답을 사용한다", async () => {
  const h = harness();
  const capture = h.camera.capture(640, 0.8);
  await new Promise(setImmediate);
  const worker = h.workers[0];
  const { data, transfer } = worker.messages[0];
  assert.equal(data.width, 360);
  assert.equal(data.height, 640);
  assert.equal(data.quality, 0.8);
  assert.equal(transfer[0], h.bitmaps[0]);
  const blob = new Blob(["jpeg"], { type: "image/jpeg" });
  worker.onmessage({ data: { id: data.id, blob } });
  assert.equal(await capture, blob);
  assert.equal(h.camera.captureBackend(), "worker");
  assert.equal(h.fallback.length, 0);
  assert.equal(h.timers.size, 0);
  h.camera.stop();
  assert.equal(worker.terminated, true);
});

test("미지원 브라우저는 기존 JPEG 경로와 크기·품질을 유지한다", async () => {
  const h = harness({ supported: false });
  assert.equal((await h.camera.capture(640, 0.8)).type, "image/jpeg");
  assert.deepEqual(h.fallback, [{ type: "image/jpeg", quality: 0.8, width: 360, height: 640 }]);
  assert.equal(h.camera.captureBackend(), "canvas");
  assert.equal(h.workers.length, 0);
});

test("VideoFrame을 지원하면 비트맵 변환 대기 없이 Worker로 바로 보낸다", async () => {
  const frames = [];
  class Frame {
    // 프레임 스냅샷 기록
    /** 카메라와 마이크로초 단위 시각을 전달받는지 확인한다. */
    constructor(video, options) { this.video = video; this.options = options; frames.push(this); }
    // 자원 해제 기록
    /** 전송 취소 경로에서도 호출 가능한 대역이다. */
    close() { this.closed = true; }
  }
  const h = harness({ videoFrameClass: Frame });
  const capture = h.camera.capture(640, 0.8);
  // Promise나 애니메이션 프레임을 기다리기 전에 전송되어야 한다.
  const { data, transfer } = h.workers[0].messages[0];
  assert.equal(data.bitmap, frames[0]);
  assert.equal(transfer[0], frames[0]);
  assert.equal(frames[0].video, h.video);
  assert.equal(frames[0].options.timestamp, 12345);
  assert.equal(h.bitmaps.length, 0);
  h.workers[0].onmessage({ data: { id: data.id, blob: new Blob(["jpeg"]) } });
  await capture;
  assert.equal(h.camera.captureBackend(), "worker_video_frame");
});

test("VideoFrame 생성 실패는 기존 ImageBitmap Worker 경로로 전환한다", async () => {
  let attempts = 0;
  class BrokenFrame {
    // 미지원 카메라 입력 재현
    /** API가 있지만 현재 영상 생성자를 지원하지 않는 경우를 검사한다. */
    constructor() { attempts++; throw new Error("unsupported video"); }
  }
  const h = harness({ videoFrameClass: BrokenFrame });
  for (let n = 0; n < 2; n++) {
    const capture = h.camera.capture(640, 0.8);
    await new Promise(setImmediate);
    const data = h.workers[0].messages[n].data;
    h.workers[0].onmessage({ data: { id: data.id, blob: new Blob(["jpeg"]) } });
    await capture;
    assert.equal(h.camera.captureBackend(), "worker");
  }
  assert.equal(attempts, 1);
  assert.equal(h.fallback.length, 0);
});

test("VideoFrame 전송 실패는 자원을 해제하고 canvas 경로를 사용한다", async () => {
  let closed = 0;
  class Frame {
    // 전송되지 못한 프레임 해제
    /** 실패 시 프레임이 메인 스레드에 남지 않는지 검사한다. */
    close() { closed++; }
  }
  const h = harness({ videoFrameClass: Frame, postError: true });
  await h.camera.capture(640, 0.8);
  assert.equal(closed, 1);
  assert.equal(h.camera.captureBackend(), "canvas");
  assert.equal(h.timers.size, 0);
});

for (const failure of ["error", "timeout", "encoding"]) {
  test(`Worker ${failure} 실패 시 정리 후 기존 경로로 전환한다`, async () => {
    const h = harness();
    const capture = h.camera.capture(640, 0.8);
    await new Promise(setImmediate);
    const worker = h.workers[0];
    if (failure === "error") worker.onerror();
    if (failure === "timeout") [...h.timers.values()][0]();
    if (failure === "encoding") worker.onmessage({ data: { id: worker.messages[0].data.id, error: "encoding failed" } });
    assert.equal((await capture).type, "image/jpeg");
    assert.equal(h.camera.captureBackend(), "canvas");
    assert.equal(worker.terminated, true);
    assert.equal(h.timers.size, 0);
    assert.equal(h.warnings.length, 1);
    await h.camera.capture(640, 0.8);
    assert.equal(h.workers.length, 1);
  });
}

test("비트맵 전송 실패 시 자원과 대기 타이머를 정리한다", async () => {
  const h = harness({ postError: true });
  assert.equal((await h.camera.capture(640, 0.8)).type, "image/jpeg");
  assert.equal(h.bitmaps[0].closed, 1);
  assert.equal(h.timers.size, 0);
  assert.equal(h.workers[0].terminated, true);
});

test("캡처 도중 카메라 종료 시 Worker 응답을 기다리지 않는다", async () => {
  const h = harness();
  const capture = h.camera.capture(640, 0.8);
  await new Promise(setImmediate);
  const worker = h.workers[0];
  h.camera.stop();
  assert.equal(await capture, null);
  assert.equal(worker.terminated, true);
  assert.equal(h.fallback.length, 0);
  assert.equal(h.timers.size, 0);
  worker.onmessage({ data: { id: 1, blob: new Blob(["late"]) } });
});

test("비트맵 생성 중에는 중복 캡처를 거부하고 종료 후 비트맵을 해제한다", async () => {
  let resolve, closed = 0;
  const h = harness({ bitmapFactory: () => new Promise((done) => { resolve = done; }) });
  const capture = h.camera.capture(640, 0.8);
  await assert.rejects(h.camera.capture(640, 0.8), /이전 캡처/);
  h.camera.stop();
  resolve({ close: () => { closed++; } });
  assert.equal(await capture, null);
  assert.equal(closed, 1);
  assert.equal(h.workers[0].messages.length, 0);
});

test("카메라 크기가 없으면 빈 캡처를 반환한다", async () => {
  const h = harness();
  h.video.videoWidth = 0;
  assert.equal(await h.camera.capture(640, 0.8), null);
  assert.equal(h.workers.length, 0);
});

// 실제 Worker 스크립트 실행 환경 생성
/** OffscreenCanvas만 대체하고 실제 메시지 처리 코드를 실행한다. */
function workerHarness(fail = false) {
  const canvases = [], replies = [], draws = [], encodes = [];
  class Canvas {
    // 캔버스 크기 보관
    /** 재사용과 화면 회전 시 크기 변경을 검사한다. */
    constructor(width, height) { this.width = width; this.height = height; canvases.push(this); }

    // 합성 호출 기록
    /** 원본 비트맵과 출력 크기를 기록한다. */
    getContext() { return { drawImage: (...args) => draws.push(args) }; }

    // JPEG 인코딩 대역
    /** 실제 Worker가 전달하는 MIME 형식과 품질을 검사한다. */
    async convertToBlob(options) {
      encodes.push(options);
      if (fail) throw new Error("encoding failed");
      return new Blob(["jpeg"], { type: options.type });
    }
  }
  const context = { self: { postMessage: (reply) => replies.push(reply) }, OffscreenCanvas: Canvas };
  vm.runInNewContext(fs.readFileSync("backend/static/js/capture-worker.js", "utf8"), context);
  return { receive: (data) => context.self.onmessage({ data }), canvases, replies, draws, encodes };
}

test("Worker는 canvas를 재사용하며 매 프레임의 비트맵을 해제한다", async () => {
  const h = workerHarness();
  let closed = 0;
  for (const [id, width, height] of [[1, 360, 640], [2, 640, 360]]) {
    await h.receive({ id, width, height, quality: 0.8, bitmap: { close: () => { closed++; } } });
  }
  assert.equal(h.canvases.length, 1);
  assert.equal(h.canvases[0].width, 640);
  assert.equal(h.canvases[0].height, 360);
  assert.equal(closed, 2);
  assert.deepEqual(h.replies.map((reply) => reply.id), [1, 2]);
  assert.equal(h.encodes[0].quality, 0.8);
  assert.equal(h.replies[0].blob.type, "image/jpeg");
});

test("Worker 인코딩 실패도 오류 응답과 비트맵 해제를 수행한다", async () => {
  const h = workerHarness(true);
  let closed = 0;
  await h.receive({ id: 1, width: 360, height: 640, quality: 0.8, bitmap: { close: () => { closed++; } } });
  assert.equal(h.replies[0].error, "encoding failed");
  assert.equal(closed, 1);
});
