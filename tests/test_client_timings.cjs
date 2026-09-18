/**
 * file_path: tests/test_client_timings.cjs
 * API에 통합한 지연 로그 수집·재시도와 비동기 마스크 그리기를 검사한다.
 * 가짜 시계와 canvas를 사용하여 서버·카메라·파일 생성 없이 실행한다.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");

// 수집기 테스트 환경 생성
/** 고정 세션의 업로드 호출과 타이머를 기록하는 실행 환경을 만든다. */
function loggerHarness(upload = async () => {}) {
  const calls = [], warnings = [], timers = new Map();
  let nextTimer = 0, nextBatch = 0;
  const context = {
    window: {}, crypto: { randomUUID: () => `batch-${++nextBatch}` },
    AbortController, setTimeout() {}, clearTimeout() {},
    setInterval: (fn, interval) => {
      assert.equal(interval, 2500); // 25건씩 보내도 10FPS 기록을 따라가야 한다.
      timers.set(++nextTimer, fn); return nextTimer;
    },
    clearInterval: (id) => timers.delete(id),
    fetch: async (path, options) => {
      const match = path.match(/^\/api\/sessions\/([^/]+)\/client-timings$/);
      assert.ok(match, path);
      assert.equal(options.method, "POST");
      assert.equal(options.headers["Content-Type"], "application/json");
      assert.equal(options.cache, "no-store");
      const id = decodeURIComponent(match[1]), batch = JSON.parse(options.body);
      calls.push({ id, batch });
      return await upload(id, batch) || { ok: true, json: async () => ({ saved_count: batch.records.length }) };
    },
  };
  vm.runInNewContext(fs.readFileSync("backend/static/js/api.js", "utf8"), context);
  return { logger: context.window.GApi.createTimings("session-A", (message) => warnings.push(message)), calls, warnings, timers };
}

// 캡처 시각을 가진 측정값 생성
/** 테스트 프레임의 캡처 시작 및 응답 수신 시각을 반환한다. */
function row(id, capture = 10, response = 110) {
  return { frame_id: id, capture_started_ms: capture, response_received_ms: response };
}

test("캡처부터 그리기까지와 이전 마스크 유지 시간을 계산한다", async () => {
  const { logger, calls, timers } = loggerHarness();
  logger.track(row(1), Promise.resolve({ status: "drawn", drawn_ms: 120 }));
  logger.track(row(2, 210, 310), Promise.resolve({ status: "drawn", drawn_ms: 330 }));
  assert.equal(await logger.finish(), true);
  const [first, second] = calls[0].batch.records;
  assert.equal(first.capture_to_overlay_ms, 110);
  assert.equal(first.response_to_overlay_ms, 10);
  assert.equal(first.overlay_interval_ms, null);
  assert.equal(second.overlay_interval_ms, 210);
  assert.equal(second.previous_overlay_age_ms, 320);
  assert.equal(calls[0].id, "session-A");
  assert.equal(timers.size, 0);
});

test("그리기 완료를 기다렸다가 마지막 로그를 저장한다", async () => {
  const { logger, calls } = loggerHarness();
  let resolve;
  logger.track(row(1), new Promise((done) => { resolve = done; }));
  const finishing = logger.finish();
  await Promise.resolve();
  assert.equal(calls.length, 0);
  resolve({ status: "superseded", drawn_ms: null });
  await finishing;
  assert.equal(calls[0].batch.records[0].overlay_status, "superseded");
  assert.equal(calls[0].batch.records[0].capture_to_overlay_ms, undefined);
});

test("25건씩 묶어 저장하고 생략된 그리기를 구별한다", async () => {
  const { logger, calls } = loggerHarness();
  for (let id = 1; id <= 26; id++) logger.track(row(id));
  await logger.finish();
  assert.deepEqual(calls.map((call) => call.batch.records.length), [25, 1]);
  assert.equal(calls[1].batch.records[0].frame_id, 26);
  assert.equal(calls[1].batch.records[0].overlay_status, "skipped");
});

test("실패한 묶음은 같은 ID로 재시도하고 종료 후 타이머를 정리한다", async () => {
  let fail = true;
  const { logger, calls, warnings, timers } = loggerHarness(async () => { if (fail) throw new Error("offline"); });
  logger.track(row(1));
  assert.equal(await logger.finish(), false);
  assert.match(warnings.at(-1), /저장 실패/);
  assert.equal(timers.size, 1);
  fail = false;
  assert.equal(await logger.flush(), true);
  assert.equal(calls[0].batch.batch_id, calls[1].batch.batch_id);
  assert.deepEqual(calls[0].batch.records, calls[1].batch.records);
  assert.equal(timers.size, 0);
});

test("통합 API에서도 서버 저장 오류를 알리고 같은 로그를 재시도한다", async () => {
  let fail = true;
  const { logger, calls, warnings } = loggerHarness(async () => fail ? {
    ok: false, status: 507,
    json: async () => ({ error: { code: "storage_failed", message: "disk full" } }),
  } : undefined);
  logger.track(row(1));
  assert.equal(await logger.finish(), false);
  assert.match(warnings.at(-1), /disk full/);
  fail = false;
  assert.equal(await logger.finish(), true);
  assert.deepEqual(calls[0].batch, calls[1].batch);
});

test("페이지는 통합 API를 앱보다 먼저 로드하고 삭제한 파일을 참조하지 않는다", () => {
  const html = fs.readFileSync("backend/static/index.html", "utf8");
  const app = fs.readFileSync("backend/static/js/app.js", "utf8");
  const scripts = [...html.matchAll(/<script src="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(scripts.length, 5);
  assert.equal(scripts[0], "/static/js/api.js?v=latency-v3-cleanup");
  assert.equal(scripts.at(-1), "/static/js/app.js?v=latency-v3-cleanup");
  assert.equal(fs.existsSync("backend/static/js/timings.js"), false);
  assert.doesNotMatch(html, /\/js\/timings\.js/);
  assert.doesNotMatch(app, /GTimings/);
  assert.match(app, /GApi\.createTimings\(/);
});

test("업로드 중 새 로그가 생겨도 동시 업로드나 누락이 없다", async () => {
  let release;
  const { logger, calls } = loggerHarness(() => new Promise((resolve) => { release = resolve; }));
  logger.track(row(1));
  await Promise.resolve();
  const first = logger.flush();
  logger.track(row(2));
  assert.equal(logger.flush(), first);
  release();
  await first;
  const second = logger.flush();
  release();
  await second;
  await logger.finish();
  assert.deepEqual(calls.flatMap((call) => call.batch.records.map((record) => record.frame_id)), [1, 2]);
});

test("오프라인 버퍼 초과를 기록하고 무한히 쌓지 않는다", async () => {
  const { logger, calls, warnings } = loggerHarness();
  for (let id = 1; id <= 502; id++) logger.track(row(id));
  await logger.finish();
  assert.equal(calls.flatMap((call) => call.batch.records).length, 500);
  assert.equal(calls[0].batch.records[0].frame_id, 3);
  assert.equal(calls[0].batch.dropped_records, 2);
  assert.match(warnings.at(-1), /2건.*누락/);
});

// 오버레이 테스트 환경 생성
/** Image 로딩과 시계를 직접 진행할 수 있는 가짜 canvas를 만든다. */
function overlayHarness(createBitmap) {
  const images = [], draws = [], pixels = [], allocations = [];
  const ctx = {
    setTransform() {}, clearRect() {}, drawImage(...args) { draws.push(args); },
    strokeRect() {}, fillRect() {}, fillText() {}, measureText() { return { width: 10 }; },
  };
  const canvas = { clientWidth: 320, clientHeight: 240, getContext: () => ctx };
  const maskCanvas = { getContext: () => ({
    createImageData: (width, height) => {
      const data = { width, height, data: new Uint8ClampedArray(width * height * 4) };
      allocations.push(data); return data;
    },
    putImageData: (image) => pixels.push(Array.from(image.data)),
  }) };
  let now = 0;
  const context = {
    window: { addEventListener() {}, devicePixelRatio: 1, createImageBitmap: createBitmap },
    document: {
      getElementById: (id) => id === "overlay" ? canvas : { videoWidth: 640, videoHeight: 480 },
      createElement: () => maskCanvas,
    },
    performance: { now: () => now },
    Image: class { constructor() { images.push(this); } },
    createImageBitmap: createBitmap, Blob, atob,
  };
  vm.runInNewContext(fs.readFileSync("backend/static/js/overlay.js", "utf8"), context);
  return { overlay: context.window.GOverlay, images, draws, pixels, allocations, tick: (time) => { now = time; } };
}

// 서버와 같은 little-endian 마스크 인코딩
/** 테스트용 길이·색 구간을 바이트 순서가 고정된 문자열로 만든다. */
function rleMask(width, height, runs) {
  const bytes = Buffer.alloc(runs.length * 4);
  runs.forEach(([length, color], index) => bytes.writeUInt32LE((length * 4) + color, index * 4));
  return { width, height, data: bytes.toString("base64") };
}

test("RLE 마스크는 디코딩 대기 없이 같은 초록·핑크·투명 픽셀을 그린다", async () => {
  const { overlay, images, draws, pixels, tick } = overlayHarness();
  tick(123);
  const drawing = overlay.draw([], { mask_rle: rleMask(3, 2, [[2, 1], [1, 2], [3, 0]]) });
  assert.equal(draws.length, 1);
  assert.equal(images.length, 0);
  assert.deepEqual(pixels[0], [0, 255, 0, 140, 0, 255, 0, 140, 255, 105, 180, 140, ...Array(12).fill(0)]);
  assert.equal((await drawing).drawn_ms, 123);
});

test("RLE 버퍼를 재사용해도 이전 마스크 픽셀이 남지 않는다", async () => {
  const { overlay, pixels, allocations } = overlayHarness();
  await overlay.draw([], { mask_rle: rleMask(2, 2, [[4, 1]]) });
  await overlay.draw([], { mask_rle: rleMask(2, 2, [[4, 0]]) });
  assert.equal(allocations.length, 1);
  assert.deepEqual(pixels[1], Array(16).fill(0));
  await overlay.draw([], { mask_rle: rleMask(4, 1, [[4, 2]]) });
  assert.equal(allocations.length, 2);
});

test("길이·색상·크기·데이터가 잘못된 RLE는 표시하지 않는다", async () => {
  const invalid = [
    rleMask(2, 2, [[3, 0]]), rleMask(2, 2, [[5, 1]]),
    rleMask(2, 2, [[4, 3]]), rleMask(2, 2, [[0, 0], [4, 1]]),
    rleMask(0, 2, [[4, 0]]), rleMask(9000, 9000, [[4, 0]]),
    { width: 2, height: 2, data: "eA==" }, { width: 2, height: 2, data: "!" },
    { width: 2, height: 2, data: "" }, { width: 2, height: 2, data: "A".repeat(21852) },
  ];
  for (const mask of invalid) {
    const { overlay, draws } = overlayHarness();
    assert.equal((await overlay.draw([], { mask_rle: mask })).status, "error");
    assert.equal(draws.length, 0);
  }
});

test("새 RLE를 그린 뒤 과거 PNG가 도착해도 화면을 덮어쓰지 않는다", async () => {
  const { overlay, images, draws } = overlayHarness();
  const previous = overlay.draw([], { mask_png: "old" });
  assert.equal((await overlay.draw([], { mask_rle: rleMask(1, 1, [[1, 2]]) })).status, "drawn");
  assert.equal((await previous).status, "superseded");
  images[0].onload();
  assert.equal(draws.length, 1);
});

test("PNG 로딩 후 canvas에 그린 시각을 반환한다", async () => {
  const { overlay, images, draws, tick } = overlayHarness();
  const drawing = overlay.draw([], { mask_png: "test" });
  assert.equal(draws.length, 0);
  tick(135);
  images[0].onload();
  const result = await drawing;
  assert.equal(result.status, "drawn");
  assert.equal(result.drawn_ms, 135);
  assert.equal(draws.length, 1);
});

test("새 마스크 및 종료 시 이전 PNG를 취소하고 늦은 콜백을 무시한다", async () => {
  const { overlay, images, draws } = overlayHarness();
  const first = overlay.draw([], { mask_png: "old" });
  const second = overlay.draw([], { mask_png: "new" });
  assert.equal((await first).status, "superseded");
  images[0].onload();
  assert.equal(draws.length, 0);
  overlay.clear();
  assert.equal((await second).status, "superseded");
  images[1].onload();
  assert.equal(draws.length, 0);
});

test("PNG 오류와 박스 그리기 완료를 구별한다", async () => {
  const { overlay, images, tick } = overlayHarness();
  const drawing = overlay.draw([], { mask_png: "broken" });
  images[0].onerror();
  assert.equal((await drawing).status, "error");
  tick(200);
  const result = await overlay.draw([{ class_name: "test", confidence: 0.9, box: { x1: 0, y1: 0, x2: 1, y2: 1 } }]);
  assert.equal(result.status, "drawn");
  assert.equal(result.drawn_ms, 200);
});

test("ImageBitmap 경로는 DOM Image 없이 그린 뒤 비트맵을 해제한다", async () => {
  let closed = 0;
  const bitmap = { close: () => { closed++; } };
  const { overlay, images, draws } = overlayHarness(async (blob) => {
    assert.equal(blob.type, "image/png");
    return bitmap;
  });
  const result = await overlay.draw([], { mask_png: "dGVzdA==" });
  assert.equal(result.status, "drawn");
  assert.equal(images.length, 0);
  assert.equal(draws[0][0], bitmap);
  assert.equal(closed, 1);
});

test("늦게 디코딩된 비트맵은 표시하지 않고 자원을 해제한다", async () => {
  let resolve, closed = 0;
  const { overlay, draws } = overlayHarness(() => new Promise((done) => { resolve = done; }));
  const drawing = overlay.draw([], { mask_png: "dGVzdA==" });
  overlay.clear();
  resolve({ close: () => { closed++; } });
  assert.equal((await drawing).status, "superseded");
  await new Promise(setImmediate);
  assert.equal(draws.length, 0);
  assert.equal(closed, 1);
});

test("비트맵 디코딩 실패 시 기존 Image 경로로 전환한다", async () => {
  const { overlay, images } = overlayHarness(async () => { throw new Error("unsupported"); });
  const drawing = overlay.draw([], { mask_png: "dGVzdA==" });
  await new Promise(setImmediate);
  assert.equal(images.length, 1);
  images[0].onload();
  assert.equal((await drawing).status, "drawn");
});

// 실제 앱 루프의 제어 가능한 실행 환경
/** 화면 요소와 업로드를 대체하여 캡처 도중 종료 및 실패 로그를 검사한다. */
async function appHarness() {
  const nodes = new Map(), tracked = [], sleeps = [], calls = [];
  const captureOptions = [];
  let sessionSettings, sessionClient;
  let now = 0, captureResolve, captureReject, uploadResolve, uploadReject;
  // 최소 화면 요소 생성
  /** 앱 이벤트를 호출하고 화면 상태를 읽을 수 있는 요소를 반환한다. */
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {
      value: id === "device-select" ? "phone" : "", dataset: {}, lastChild: {}, handlers: {},
      classList: { add() {}, remove() {}, toggle() {} }, options: [],
      querySelector() { return this; }, querySelectorAll() { return []; },
      addEventListener(name, fn) { this.handlers[name] = fn; }, scrollIntoView() {},
    });
    return nodes.get(id);
  }
  const logger = {
    track: (record, drawing) => { tracked.push({ record, drawing }); calls.push("track"); },
    finish: async () => { calls.push("finish"); return true; },
  };
  const context = {
    window: { scrollTo() {}, addEventListener() {} },
    document: { getElementById: node, querySelector: node, hidden: false, visibilityState: "visible", addEventListener() {} },
    performance: { now: () => now }, Date: { now: () => 1000 + now },
    localStorage: { getItem: () => null, setItem() {} }, navigator: {}, screen: {},
    setInterval() {}, setTimeout: (fn, ms) => { sleeps.push({ fn, ms }); },
    GCamera: {
      setOnEnded() {}, start: async () => {}, active: () => true,
      captureBackend: () => "worker",
      capture: (maxSide, quality) => {
        captureOptions.push({ maxSide, quality });
        return new Promise((resolve, reject) => { captureResolve = resolve; captureReject = reject; });
      },
    },
    GRecorder: { start() {}, stop: async () => null, active: () => true },
    GOverlay: {
      clear: () => calls.push("clear"), describeDetection: () => ({}),
      draw: () => { calls.push("draw"); return Promise.resolve({ status: "drawn", drawn_ms: now }); },
    },
    GApi: {
      createTimings: (id) => { assert.equal(id, "session-A"); return logger; },
      health: async () => ({ storage_writable: true }),
      models: async () => ({ models: [{ id: "traffic-mock", mode: "traffic", available: true }] }),
      createSession: async (body) => {
        sessionSettings = body.settings;
        sessionClient = body.client;
        return { session_id: "session-A" };
      },
      uploadFrame: () => {
        calls.push("upload");
        return new Promise((resolve, reject) => { uploadResolve = resolve; uploadReject = reject; });
      },
      stopSession: async () => { calls.push("stop"); return { session: {} }; },
    },
  };
  vm.runInNewContext(fs.readFileSync("backend/static/js/app.js", "utf8"), context);
  await Promise.resolve();
  await node("btn-camera").handlers.click();
  await node("btn-start").handlers.click();
  return {
    tracked, calls, sleeps, node, captureOptions, sessionSettings, sessionClient,
    tick: (time) => { now = time; },
    captured: (blob = { size: 42000 }) => captureResolve(blob),
    captureFailed: () => captureReject(new Error("camera error")),
    responded: () => uploadResolve({ frame_id: 1, saved: true, timing: {} }),
    failed: () => uploadReject({ code: "timeout", status: 0, message: "timeout" }),
    stop: () => node("btn-stop").handlers.click(),
  };
}

test("640px 전송 설정을 실제 캡처와 세션 기록에 동일하게 적용한다", async () => {
  const app = await appHarness();
  assert.deepEqual(app.captureOptions, [{ maxSide: 640, quality: 0.8 }]);
  assert.equal(app.sessionSettings.image_max_side, 640);
  assert.equal(app.sessionSettings.jpeg_quality, 0.8);
  assert.equal(app.sessionSettings.target_fps, 10);
  assert.equal(app.sessionSettings.confidence, 0.4);
  assert.equal(app.sessionClient.app_version, "latency-v3-cleanup");
  const stopping = app.stop();
  app.captured();
  await stopping;
});

test("캡처 도중 종료해도 추가 전송 없이 취소 로그를 먼저 저장한다", async () => {
  const app = await appHarness();
  const stopping = app.stop();
  app.tick(25);
  app.captured();
  await stopping;
  assert.equal(app.calls.includes("upload"), false);
  assert.equal(app.tracked[0].record.status, "cancelled");
  assert.equal(app.tracked[0].record.capture_ms, 25);
  assert.equal(app.tracked[0].record.captured_at_ms, 1000);
  assert.ok(app.calls.indexOf("track") < app.calls.indexOf("finish"));
  assert.ok(app.calls.indexOf("finish") < app.calls.indexOf("stop"));
  assert.equal(app.calls.filter((call) => call === "clear").length, 1);
});

test("수신·그리기와 10FPS 제한 대기 시간을 분리한다", async () => {
  const app = await appHarness();
  app.tick(20);
  app.captured();
  await new Promise(setImmediate);
  app.tick(70);
  app.responded();
  await new Promise(setImmediate);
  assert.equal(app.sleeps[0].ms, 30);
  assert.equal(app.node("m-rtt").textContent, "70");
  const stopping = app.stop();
  app.tick(105);
  app.sleeps[0].fn();
  await stopping;
  const row = app.tracked[0].record;
  assert.equal(row.capture_ms, 20);
  assert.equal(row.request_ms, 50);
  assert.equal(row.throttle_wait_ms, 35);
  assert.equal(row.status, "ok");
  assert.equal(row.recording_active, true);
  assert.equal(row.capture_backend, "worker");
});

test("응답 대기 중에는 캡처를 쌓지 않고 느린 응답 뒤 추가 대기도 없다", async () => {
  const app = await appHarness();
  app.tick(20);
  app.captured();
  await new Promise(setImmediate);
  assert.equal(app.captureOptions.length, 1);
  assert.equal(app.calls.filter((call) => call === "upload").length, 1);
  app.tick(250);
  app.responded();
  await new Promise(setImmediate);
  assert.equal(app.sleeps.length, 0);
  assert.equal(app.captureOptions.length, 2);
  const stopping = app.stop();
  app.captured();
  await stopping;
  assert.equal(app.calls.filter((call) => call === "upload").length, 1);
});

test("종료 중 도착한 응답은 그리지 않고 마지막 요청 시간을 기록한다", async () => {
  const app = await appHarness();
  app.tick(20);
  app.captured();
  await new Promise(setImmediate);
  const stopping = app.stop();
  app.tick(220);
  app.responded();
  await stopping;
  assert.equal(app.calls.includes("draw"), false);
  assert.equal(app.tracked[0].record.status, "cancelled");
  assert.equal(app.tracked[0].record.request_ms, 200);
});

test("시간 초과와 오류 재시도 대기 시간을 남긴다", async () => {
  const app = await appHarness();
  app.tick(20);
  app.captured();
  await new Promise(setImmediate);
  app.tick(5020);
  app.failed();
  await new Promise(setImmediate);
  assert.equal(app.sleeps[0].ms, 500);
  const stopping = app.stop();
  app.tick(5520);
  app.sleeps[0].fn();
  await stopping;
  const row = app.tracked[0].record;
  assert.equal(row.status, "error");
  assert.equal(row.error_code, "timeout");
  assert.equal(row.request_ms, 5000);
  assert.equal(row.retry_wait_ms, 500);
  assert.equal(row.response_received_ms, undefined);
});

test("카메라 캡처 실패도 기록하고 정상 종료할 수 있다", async () => {
  const app = await appHarness();
  app.tick(30);
  app.captureFailed();
  await new Promise(setImmediate);
  const stopping = app.stop();
  app.tick(530);
  app.sleeps[0].fn();
  await stopping;
  const row = app.tracked[0].record;
  assert.equal(row.error_code, "capture_failed");
  assert.equal(row.capture_ms, 30);
  assert.equal(row.request_ms, undefined);
  assert.equal(app.calls.includes("upload"), false);
});

test("JPEG가 생성되지 않은 경우도 빈 캡처로 구별한다", async () => {
  const app = await appHarness();
  app.tick(30);
  app.captured(null);
  await new Promise(setImmediate);
  assert.equal(app.sleeps[0].ms, 100);
  const stopping = app.stop();
  app.tick(130);
  app.sleeps[0].fn();
  await stopping;
  const row = app.tracked[0].record;
  assert.equal(row.error_code, "capture_empty");
  assert.equal(row.jpeg_bytes, 0);
  assert.equal(row.retry_wait_ms, 100);
});
