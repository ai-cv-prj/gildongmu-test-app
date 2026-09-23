/**
 * 파일 경로: tests/test_client_timings.cjs
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
  assert.equal(scripts.length, 8);
  assert.equal(scripts[0], "/static/js/api.js?v=latency-v3-cleanup");
  assert.equal(scripts.at(-1), "/static/js/app.js?v=traffic-display-v14");
  assert.ok(scripts.indexOf("/static/js/tts.js?v=walking-audio-v5") >= 0);
  assert.ok(scripts.indexOf("/static/js/tts.js?v=walking-audio-v5") < scripts.length - 1);
  assert.ok(scripts.includes("/static/js/guidance.js?v=walking-audio-v5"));
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
async function appHarness(mode = "walking", { start = true, prefs = {}, storageAvailable = true, deferSession = false } = {}) {
  const nodes = new Map(), tracked = [], sleeps = [], calls = [], visibilityHandlers = {};
  const media = require("./helpers/audio.cjs").audioHarness({ onPlay: () => calls.push("play") });
  const captureOptions = [];
  let sessionSettings, sessionClient, resolveSession;
  let storedPrefs = JSON.stringify({ mode, ...prefs });
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
    window: {
      scrollTo() {}, addEventListener() {},
      Audio: media.Audio,
    },
    document: { getElementById: node, querySelector: node, hidden: false, visibilityState: "visible", addEventListener(name, fn) { visibilityHandlers[name] = fn; } },
    performance: { now: () => now, timeOrigin: 1000 }, Date: { now: () => 1000 + now },
    localStorage: {
      getItem: () => { if (!storageAvailable) throw new Error("unavailable"); return storedPrefs; },
      setItem: (key, value) => { if (!storageAvailable) throw new Error("unavailable"); storedPrefs = value; },
    }, navigator: {}, screen: {},
    setInterval() {},
    setTimeout: (fn, ms) => { const timer = { fn, ms }; sleeps.push(timer); return timer; },
    clearTimeout: (timer) => { const index = sleeps.indexOf(timer); if (index >= 0) sleeps.splice(index, 1); },
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
      clear: () => calls.push("clear"), describeDetection: () => ({}), describeCrosswalkEvent: () => "",
      draw: () => { calls.push("draw"); return Promise.resolve({ status: "drawn", drawn_ms: now }); },
    },
    GApi: {
      createTimings: (id) => { assert.equal(id, "session-A"); return logger; },
      health: async () => ({ storage_writable: true }),
      models: async () => ({ models: [{ id: `${mode}-mock`, mode, available: true }] }),
      createSession: async (body) => {
        calls.push("create-session");
        sessionSettings = body.settings;
        sessionClient = body.client;
        if (deferSession) return new Promise(resolve => { resolveSession = resolve; });
        return { session_id: "session-A" };
      },
      uploadFrame: () => {
        calls.push("upload");
        return new Promise((resolve, reject) => { uploadResolve = resolve; uploadReject = reject; });
      },
      stopSession: async () => { calls.push("stop"); return { session: {} }; },
    },
  };
  vm.createContext(context);
  for (const name of ["tts", "guidance"]) {
    vm.runInContext(fs.readFileSync(`backend/static/js/${name}.js`, "utf8"), context);
  }
  context.GTts = context.window.GTts;
  context.GGuidance = context.window.GGuidance;
  vm.runInContext(fs.readFileSync("backend/static/js/app.js", "utf8"), context);
  await Promise.resolve();
  if (start) {
    await node("btn-camera").handlers.click();
    await node("btn-start").handlers.click();
    // 도보 기본 시작 음성을 끝내 일반 프레임 테스트가 음성 로딩 타이머와 섞이지 않게 한다.
    if (mode === "walking" && media.plays[0]) {
      media.plays[0].start();
      media.plays[0].end();
    }
  }
  return {
    tracked, calls, sleeps, node, captureOptions, sessionSettings, sessionClient, played: media.plays,
    prefs: () => JSON.parse(storedPrefs), resolveSession: () => resolveSession({ session_id: "session-A" }),
    hide: () => { context.document.hidden = true; visibilityHandlers.visibilitychange(); },
    tick: (time) => { now = time; },
    captured: (blob = { size: 42000 }) => captureResolve(blob),
    captureFailed: () => captureReject(new Error("camera error")),
    responded: (result = {}) => uploadResolve({ session_id: "session-A", frame_id: 1, saved: true, timing: {}, ...result }),
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
  assert.equal(app.sessionSettings.confidence, 0.25);
  assert.equal(app.sessionClient.app_version, "walking-risk-v1");
  const stopping = app.stop();
  app.captured();
  await stopping;
});

test("신호등은 960px·0.25를 기록하고 실제 전송도 5FPS로 제한한다", async () => {
  const app = await appHarness("traffic");
  assert.deepEqual(app.captureOptions, [{ maxSide: 960, quality: 0.8 }]);
  assert.equal(app.sessionSettings.image_max_side, 960);
  assert.equal(app.sessionSettings.target_fps, 5);
  assert.equal(app.sessionSettings.confidence, 0.25);
  app.tick(20);
  app.captured();
  await new Promise(setImmediate);
  app.tick(70);
  app.responded();
  await new Promise(setImmediate);
  assert.equal(app.sleeps.at(-1).ms, 130);
  const stopping = app.stop();
  app.tick(200);
  app.sleeps.at(-1).fn();
  await stopping;
});

test("버스도 640px·10FPS·0.4 설정을 유지한다", async () => {
  const app = await appHarness("bus");
  assert.deepEqual(app.captureOptions, [{ maxSide: 640, quality: 0.8 }]);
  assert.equal(app.sessionSettings.target_fps, 10);
  assert.equal(app.sessionSettings.confidence, 0.4);
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

// 실제 앱 이벤트와 프레임 루프를 통해 안내 모듈 연결을 검증한다.
test("신호등 테스트 시작 클릭에서 자동 재생하고 최초 초록불에 다음 신호 대기를 안내한다", async () => {
  const app = await appHarness("traffic");
  assert.equal(app.node("guidance-panel").hidden, false);
  assert.equal(app.played.length, 1);
  assert.ok(app.calls.indexOf("play") < app.calls.indexOf("create-session"));
  assert.equal(app.played[0].src, "/static/audio/ko-v1/startup.mp3");
  assert.equal(app.node("btn-mute").textContent, "음성 끄기");
  app.played[0].start(); app.played[0].end();
  for (let i = 0; i < 3; i++) {
    app.captured();
    await new Promise(setImmediate);
    app.tick(i * 200 + 50);
    app.responded({ frame_id: i + 1, detections: [{ track_id: 1 }],
      event: { type: "traffic_signal", signal_state: "green", selected_detection_index: 0 } });
    await new Promise(setImmediate);
    if (i < 2) {
      app.tick((i + 1) * 200);
      app.sleeps.findLast(s => s.ms === 150).fn();
      await new Promise(setImmediate);
    }
  }
  assert.equal(app.played.at(-1).src, "/static/audio/ko-v1/green-initial-wait.mp3");
  const stopping = app.stop();
  assert.equal(app.node("btn-mute").disabled, false);
  app.tick(600); app.sleeps.findLast(s => s.ms === 150).fn();
  await stopping;
  assert.equal(app.node("guidance-panel").hidden, false);
});

test("화면 숨김은 음성을 종료하고 늦게 도착한 신호로 재개하지 않는다", async () => {
  const app = await appHarness("traffic");
  app.captured(); await new Promise(setImmediate);
  app.hide();
  assert.equal(app.node("btn-mute").disabled, false);
  assert.match(app.node("guidance-text").textContent, /화면이 숨겨져/);
  const stopping = app.stop();
  app.responded({ detections: [{ track_id: 1 }],
    event: { type: "traffic_signal", signal_state: "green", selected_detection_index: 0 } });
  await stopping;
  assert.equal(app.played.length, 1);
});

test("음원 로딩 실패는 안내 상태를 종료하고 다시 시작할 수 있게 한다", async () => {
  const app = await appHarness("traffic");
  app.node("btn-mute").handlers.click();
  app.node("btn-mute").handlers.click();
  app.played.at(-1).error(2);
  assert.match(app.node("guidance-text").textContent, /서버 연결/);
  assert.equal(app.node("btn-mute").disabled, false);
  assert.equal(app.node("btn-mute").textContent, "음성 끄기");
  app.node("btn-mute").handlers.click();
  app.node("btn-mute").handlers.click();
  assert.equal(app.played.length, 3);
  const stopping = app.stop(); app.captured(); await stopping;
});

test("도보 장애물 테스트에서도 음성 패널을 표시하고 시작 안내를 재생한다", async () => {
  const app = await appHarness("walking");
  assert.equal(app.node("guidance-panel").hidden, false);
  assert.equal(app.node("btn-mute").disabled, false);
  assert.equal(app.played.length, 1);
  assert.equal(app.played[0].src, "/static/audio/ko-v1/walking-startup.mp3");
  const stopping = app.stop(); app.captured(); await stopping;
});

test("카메라와 세션 없이 음성 확인을 재생하고 실제 재생 상태를 표시한다", async () => {
  const app = await appHarness("traffic", { start: false });
  assert.equal(app.node("guidance-panel").hidden, false);
  assert.equal(app.node("btn-mute").disabled, false);
  app.node("btn-sound-check").handlers.click();
  assert.equal(app.played[0].src, "/static/audio/ko-v1/sound-check.mp3");
  assert.equal(app.calls.includes("create-session"), false);
  assert.equal(app.captureOptions.length, 0);
  app.played[0].start();
  assert.match(app.node("tts-status").textContent, /재생 중/);
  app.played[0].end();
  assert.match(app.node("tts-status").textContent, /끝났습니다/);
});

test("기본 음성 안내를 끄고 다시 켤 수 있다", async () => {
  const app = await appHarness("traffic");
  app.node("btn-mute").handlers.click();
  assert.equal(app.node("btn-mute").textContent, "음성 켜기");
  const count = app.played.length;
  app.node("btn-mute").handlers.click();
  assert.equal(app.played.length, count + 1);
  assert.equal(app.node("btn-mute").textContent, "음성 끄기");
  const stopping = app.stop(); app.captured(); await stopping;
});

test("실제 프레임 루프에서도 대상 교체로 같은 색 음원을 반복하지 않는다", async () => {
  const app = await appHarness("traffic");
  app.played[0].start(); app.played[0].end();
  const frames = [
    ...Array(3).fill({ color: "red", target: 1 }),
    ...Array(3).fill({ color: "red", target: 2 }),
    ...Array(3).fill({ color: "green", target: 2 }),
    ...Array(3).fill({ color: "green", target: 1 }),
  ];
  for (let i = 0; i < frames.length; i++) {
    const { color, target } = frames[i];
    app.captured();
    await new Promise(setImmediate);
    app.tick(i * 200 + 50);
    app.responded({ frame_id: i + 1, detections: [{ track_id: target }],
      event: { type: "traffic_signal", signal_state: color, selected_detection_index: 0 } });
    await new Promise(setImmediate);
    if (i === 2) app.played[1].start();
    if (i < frames.length - 1) {
      app.tick((i + 1) * 200);
      app.sleeps.findLast(s => s.ms === 150).fn();
      await new Promise(setImmediate);
    }
  }
  // 빨간불 안내가 아직 끝나지 않았으므로 초록 전환은 대기한다.
  assert.deepEqual(app.played.map(p => p.src), [
    "/static/audio/ko-v1/startup.mp3", "/static/audio/ko-v1/red.mp3",
  ]);
  app.played[1].end();
  assert.equal(app.played[2].src, "/static/audio/ko-v1/green-changed.mp3");
  assert.match(app.node("guidance-text").textContent, /초록불입니다/);
  const stopping = app.stop();
  app.tick(frames.length * 200); app.sleeps.findLast(s => s.ms === 150).fn();
  await stopping;
});

test("대상 재확인 설명은 2.5초 보관·교체 보류 상태를 표시한다", () => {
  const { overlay } = overlayHarness();
  for (const reason of ["waiting_for_target_reacquisition", "waiting_for_target_hold"]) {
    assert.match(overlay.describeCrosswalkEvent({ crosswalk_diagnostics: { connection_status: reason, candidate_count: 0 } }), /기존/);
  }
});

test("카메라와 테스트 없이 음성 설정을 바꾸고 선택만으로는 재생하지 않는다", async () => {
  const app = await appHarness("traffic", { start: false });
  assert.equal(app.node("btn-mute").disabled, false);
  assert.equal(app.node("btn-mute").textContent, "음성 끄기");
  app.node("btn-mute").handlers.click();
  assert.equal(app.node("btn-mute").textContent, "음성 켜기");
  assert.equal(app.prefs().voiceEnabled, false);
  app.node("btn-mute").handlers.click();
  assert.equal(app.prefs().voiceEnabled, true);
  assert.match(app.node("guidance-text").textContent, /테스트를 시작하면/);
  assert.equal(app.played.length, 0);
  assert.equal(app.calls.includes("create-session"), false);
});

test("테스트 전에 꺼둔 설정은 카메라 시작과 다음 테스트에도 유지한다", async () => {
  const app = await appHarness("traffic", { start: false });
  app.node("btn-mute").handlers.click();
  await app.node("btn-camera").handlers.click();
  await app.node("btn-start").handlers.click();
  assert.equal(app.played.length, 0);
  assert.equal(app.node("btn-mute").textContent, "음성 켜기");
  let stopping = app.stop(); app.captured(); await stopping;
  assert.equal(app.node("btn-mute").disabled, false);
  await app.node("btn-start").handlers.click();
  assert.equal(app.played.length, 0);
  stopping = app.stop(); app.captured(); await stopping;
});

test("저장된 끄기 설정을 새 페이지에 복원하고 테스트 중에는 바로 켤 수 있다", async () => {
  const first = await appHarness("traffic", { start: false });
  first.node("btn-mute").handlers.click();
  const app = await appHarness("traffic", { prefs: first.prefs() });
  assert.equal(app.played.length, 0);
  assert.equal(app.node("btn-mute").textContent, "음성 켜기");
  app.node("btn-mute").handlers.click();
  assert.equal(app.played.length, 1);
  assert.equal(app.prefs().voiceEnabled, true);
  const stopping = app.stop(); app.captured(); await stopping;
});

test("미리 켜둔 설정은 테스트 시작 클릭 안에서 처음 재생한다", async () => {
  const app = await appHarness("traffic", { start: false, prefs: { voiceEnabled: false } });
  app.node("btn-mute").handlers.click();
  assert.equal(app.played.length, 0);
  await app.node("btn-camera").handlers.click();
  await app.node("btn-start").handlers.click();
  assert.equal(app.played.length, 1);
  assert.ok(app.calls.indexOf("play") < app.calls.indexOf("create-session"));
  const stopping = app.stop(); app.captured(); await stopping;
});

test("세션 생성 응답을 기다리는 동안 꺼도 늦은 응답이 음성을 켜지 않는다", async () => {
  const app = await appHarness("traffic", { start: false, deferSession: true });
  await app.node("btn-camera").handlers.click();
  const starting = app.node("btn-start").handlers.click();
  assert.equal(app.played.length, 1);
  app.node("btn-mute").handlers.click();
  app.resolveSession(); await starting;
  assert.equal(app.played.length, 1);
  assert.equal(app.node("btn-mute").textContent, "음성 켜기");
  assert.equal(app.prefs().voiceEnabled, false);
  const stopping = app.stop(); app.captured(); await stopping;
});

test("설정 저장이 차단돼도 현재 페이지의 음성 끄기를 적용한다", async () => {
  const app = await appHarness("traffic", { start: false, storageAvailable: false });
  app.node("btn-mute").handlers.click();
  await app.node("btn-camera").handlers.click();
  await app.node("btn-start").handlers.click();
  assert.equal(app.played.length, 0);
  const stopping = app.stop(); app.captured(); await stopping;
});

test("음성 확인도 시작 안내를 끊지 않고 종료 후 순서대로 재생한다", async () => {
  const app = await appHarness("traffic");
  app.played[0].start();
  app.node("btn-sound-check").handlers.click();
  assert.equal(app.played.length, 1);
  app.played[0].end();
  assert.equal(app.played[1].src, "/static/audio/ko-v1/sound-check.mp3");
  assert.equal(app.node("btn-mute").textContent, "음성 끄기");
  app.played[1].start(); app.played[1].end();
  const stopping = app.stop(); app.captured(); await stopping;
});

for (const action of ["mute", "stop"]) {
  test(`${action}: 앱에서 중단하면 대기 중인 후속 음성도 지운다`, async () => {
    const app = await appHarness("traffic");
    app.played[0].start();
    app.node("btn-sound-check").handlers.click();
    assert.equal(app.played.length, 1);
    if (action === "mute") app.node("btn-mute").handlers.click();
    const stopping = app.stop(); app.captured(); await stopping;
    app.played[0].end();
    await new Promise(setImmediate);
    assert.equal(app.played.length, 1);
  });
}

test("신호등 미선택·후보·확정 박스에 추적 ID를 표시한다", () => {
  const { overlay } = overlayHarness();
  for (const selection_status of ["unselected", "candidate", "selected"]) {
    const style = overlay.describeDetection({ class_name: "pedestrian_signal", confidence: .9,
      track_id: 7, extra: { selection_status, signal_state: "red", color_confidence: .95 } });
    assert.match(style.label, /ID 7/);
  }
});
