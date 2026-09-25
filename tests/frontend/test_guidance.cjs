/*
file_path: tests/frontend/test_guidance.cjs

브라우저 기능 검증을 위한 테스트 또는 보조 코드를 담은 모듈이다.
*/
/** 신호 안내 정책과 음원 재생 연결의 시간·대상·취소 경계를 검증한다. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");

function harness({ deferStartup = false } = {}) {
  let time = 0, id = 0;
  const speech = [], updates = [], cancellations = [];
  const context = { window: {} };
  vm.runInNewContext(fs.readFileSync("backend/static/js/guidance.js", "utf8"), context);
  const policy = context.window.GGuidance.create({
    now: () => time, onChange: (state) => updates.push(state),
    player: {
      speak: (text, deadline, options) => {
        speech.push({ text, deadline, options });
        if (!deferStartup) options?.onEnd();
      },
      cancel: () => cancellations.push(time),
    },
  });
  function frame(color, at, target = 1, options = {}) {
    time = options.receivedAt ?? at;
    policy.accept({
      session_id: options.session ?? "A", frame_id: options.frameId ?? ++id,
      detections: [{ track_id: target }],
      event: { type: "traffic_signal", signal_state: color, selected_detection_index: options.unselected ? null : 0 },
    }, at);
  }
  function stable(color, at, target = 1) {
    for (let i = 0; i < 3; i++) frame(color, at + i * 200, target);
  }
  return { policy, frame, stable, speech, updates, cancellations,
    tick(at) { time = at; policy.tick(); }, setTime(at) { time = at; },
    messages: () => speech.map(item => item.text),
  };
}

test("음성 안내를 켜기 전에는 검출 결과가 있어도 재생하지 않는다", () => {
  const h = harness();
  h.stable("green", 0);
  assert.deepEqual(h.messages(), []);
});

test("시작 음성이 끝나기 전에는 추론이나 감지 공백으로 첫 재생을 취소하지 않는다", () => {
  const h = harness({ deferStartup: true }); h.policy.start("A");
  const before = h.cancellations.length;
  h.stable("red", 0); h.tick(2000);
  assert.equal(h.cancellations.length, before);
  assert.equal(h.speech.length, 1);
  h.speech[0].options.onEnd();
  h.frame("red", 1800, 1, { receivedAt: 2100 });
  assert.equal(h.speech.length, 1);
  h.stable("green", 2200);
  assert.equal(h.messages().at(-1), "초록불입니다. 다음 초록 신호를 기다려 주세요.");
});

test("처음 본 초록불은 다음 초록 신호 대기를 안내하고 같은 색은 반복하지 않는다", () => {
  const h = harness(); h.policy.start("A");
  h.stable("green", 0); h.stable("green", 600);
  assert.deepEqual(h.messages(), ["신호 안내를 시작합니다.", "초록불입니다. 다음 초록 신호를 기다려 주세요."]);
});

for (const kind of ["무응답", "색상 미확인", "대상 미선택", "짧은 검출"]) {
  test(`최초 신호 확인 전 ${kind} 상태에서는 소실 음성을 내지 않는다`, () => {
    const h = harness(); h.policy.start("A");
    if (kind === "색상 미확인") h.stable("unknown", 0);
    if (kind === "대상 미선택") h.frame("red", 0, 1, { unselected: true });
    if (kind === "짧은 검출") {
      h.frame("red", 0); h.frame("red", 200); h.frame("unknown", 400);
    }
    h.tick(2000); h.tick(10000);
    assert.deepEqual(h.messages(), ["신호 안내를 시작합니다."]);
  });
}

test("안내를 다시 켜면 신호 확인 이력을 지우고 새로 확인한 이후에만 소실을 알린다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0);
  h.tick(2400);
  const missingCount = () => h.messages().filter(s => s === "신호를 확인할 수 없습니다.").length;
  assert.equal(missingCount(), 1);
  h.setTime(2600); h.policy.start("B"); h.tick(10000);
  assert.equal(missingCount(), 1);
  for (const at of [10200, 10400, 10600]) h.frame("green", at, 1, { session: "B" });
  h.tick(12100); h.tick(20000);
  assert.equal(missingCount(), 2);
});

test("같은 대상의 연속 빨강·초록·빨강 변화를 안내한다", () => {
  const h = harness(); h.policy.start("A");
  h.stable("red", 0); h.stable("green", 600); h.stable("red", 1200);
  assert.deepEqual(h.messages().slice(1), ["빨간불입니다.", "초록불로 바뀌었습니다.", "빨간불로 바뀌었습니다."]);
});

for (const color of ["red", "green"]) {
  test(`${color}: 추적 대상이 여러 번 바뀌어도 같은 색은 한 번만 안내한다`, () => {
    const h = harness(); h.policy.start("A");
    h.stable(color, 0, 1);
    h.stable(color, 600, 2);
    h.stable(color, 1200, 1);
    h.stable(color, 1800, 3);
    assert.equal(h.speech.length, 2);
    assert.match(h.updates.at(-1).text, color === "red" ? /빨간불입니다/ : /초록불입니다/);
  });
}

for (const kind of ["unknown", "unselected", "gap", "error", "frame-gap"]) {
  test(`소실 안내 없는 ${kind} 이후 같은 색을 재확인해도 반복하지 않는다`, () => {
    const h = harness(); h.policy.start("A"); h.stable("red", 0);
    if (kind === "unknown") h.frame("unknown", 600);
    if (kind === "unselected") h.frame("red", 600, 1, { unselected: true });
    if (kind === "error") h.policy.interrupt();
    if (kind === "frame-gap") {
      for (let i = 0; i < 3; i++) h.frame("red", 800 + i * 200, 1, { frameId: 9 + i });
    } else {
      h.stable("red", kind === "gap" ? 2000 : 800);
    }
    assert.equal(h.messages().filter(text => text === "빨간불입니다.").length, 1);
    assert.equal(h.updates.at(-1).text, "빨간불입니다.");
  });
}

test("다른 색을 확정한 뒤 원래 색으로 돌아오면 각각 한 번씩 안내한다", () => {
  const h = harness(); h.policy.start("A");
  h.stable("red", 0, 1); h.stable("red", 600, 2);
  h.stable("green", 1200, 2); h.stable("green", 1800, 3);
  h.stable("red", 2400, 3); h.stable("red", 3000, 1);
  assert.deepEqual(h.messages().slice(1), ["빨간불입니다.", "초록불로 바뀌었습니다.", "빨간불로 바뀌었습니다."]);
});

test("대상 변경 후 다른 색은 한 번 알리되 전환 시점을 확인했다고 하지 않는다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0, 1);
  h.stable("green", 600, 2); h.stable("green", 1200, 1);
  assert.deepEqual(h.messages().slice(1), ["빨간불입니다.", "초록불입니다."]);
});

test("확정되지 않은 반대 색과 대상 교체는 같은 색의 반복 억제를 해제하지 않는다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0);
  h.frame("green", 600, 2); h.frame("green", 800, 2);
  h.stable("red", 1000, 3);
  assert.deepEqual(h.messages(), ["신호 안내를 시작합니다.", "빨간불입니다."]);
});

test("교체된 대상들의 반대 색 후보를 합쳐 전환으로 확정하지 않는다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0);
  h.frame("green", 600, 2); h.frame("green", 800, 3); h.frame("green", 1000, 2);
  h.stable("red", 1200, 1);
  assert.deepEqual(h.messages(), ["신호 안내를 시작합니다.", "빨간불입니다."]);
});

test("소실 안내 후 같은 색을 다시 읽고 다음 소실 구간도 한 번 안내한다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0);
  h.tick(2400); h.stable("red", 2600); h.tick(5000); h.tick(10000);
  assert.equal(h.messages().filter(text => text === "빨간불입니다.").length, 2);
  assert.equal(h.messages().filter(text => text === "신호를 확인할 수 없습니다.").length, 2);
});

for (const color of ["red", "green"]) {
  for (const recoveredTarget of [1, 2]) {
    test(`소실 안내 후 ${color} 재확인은 대상 ${recoveredTarget}에서 한 번 안내한다`, () => {
      const h = harness(); h.policy.start("A"); h.stable(color, 0);
      h.tick(2400); h.tick(10000);
      assert.equal(h.messages().at(-1), "신호를 확인할 수 없습니다.");
      // 짧은 재검출로 재안내 기회를 소모하지 않는다.
      h.frame(color, 10200, recoveredTarget);
      h.frame("unknown", 10400, recoveredTarget);
      assert.equal(h.speech.length, 3);
      h.frame(color, 10600, recoveredTarget);
      h.frame(color, 10800, recoveredTarget);
      assert.equal(h.speech.length, 3);
      h.frame(color, 11000, recoveredTarget);
      assert.equal(h.messages().at(-1), color === "red" ? "빨간불입니다."
        : "초록불입니다.");
      h.stable(color, 11200, recoveredTarget);
      h.stable(color, 11800, 3);
      assert.equal(h.speech.length, 4);
    });
  }
}

for (const session of ["A", "B"]) {
  test(`안내 재시작 시 같은 색도 다시 안내한다: 세션 ${session}`, () => {
    const h = harness(); h.policy.start("A"); h.stable("red", 0);
    h.policy.stop(); h.setTime(1000); h.policy.start(session);
    for (const at of [1000, 1200, 1400]) h.frame("red", at, 1, { session });
    assert.equal(h.messages().filter(text => text === "빨간불입니다.").length, 2);
  });
}

test("색상은 세 프레임과 최소 관측 시간을 모두 충족해야 확정한다", () => {
  const h = harness(); h.policy.start("A");
  h.frame("red", 0); h.frame("red", 10); h.frame("red", 20);
  assert.equal(h.speech.length, 1);
  h.frame("red", 400);
  assert.equal(h.messages().at(-1), "빨간불입니다.");
});

test("한 프레임의 초록 오분류는 전환으로 발화하지 않고 기존 음성을 유지한다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0);
  const before = h.cancellations.length;
  h.frame("green", 600); h.stable("red", 800);
  assert.equal(h.cancellations.length, before);
  assert.equal(h.speech.length, 2);
});

for (const kind of ["unknown", "unselected", "gap", "target", "error", "frame-gap"]) {
  test(`${kind} 이후 초록불을 신규 전환으로 오인하지 않는다`, () => {
    const h = harness(); h.policy.start("A"); h.stable("red", 0);
    if (kind === "unknown") h.frame("unknown", 600);
    if (kind === "unselected") h.frame("green", 600, 1, { unselected: true });
    if (kind === "error") h.policy.interrupt();
    if (kind === "frame-gap") h.frame("green", 600, 1, { frameId: 4 + 5 });
    if (kind === "frame-gap") {
      h.frame("green", 800, 1, { frameId: 10 });
      h.frame("green", 1000, 1, { frameId: 11 });
    } else h.stable("green", kind === "gap" ? 2000 : 800, kind === "target" ? 2 : 1);
    assert.equal(h.messages().at(-1), "초록불입니다.");
  });
}

test("통신이 멈춰도 기존 음성은 유지하고 판단 불가를 한 번 요청한다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0);
  const before = h.cancellations.length;
  h.tick(1500);
  assert.equal(h.cancellations.length, before);
  h.tick(1900); h.tick(2399);
  assert.equal(h.messages().filter(s => s === "신호를 확인할 수 없습니다.").length, 0);
  h.tick(2400);
  assert.equal(h.messages().at(-1), "신호를 확인할 수 없습니다.");
  h.tick(10000);
  assert.equal(h.messages().filter(s => s === "신호를 확인할 수 없습니다.").length, 1);
  h.stable("green", 10200);
  assert.equal(h.messages().at(-1), "초록불입니다.");
  h.tick(12100); h.tick(20000);
  assert.equal(h.messages().filter(s => s === "신호를 확인할 수 없습니다.").length, 2);
});

for (const kind of ["unknown", "unselected", "target", "error", "frame-gap"]) {
  test(`${kind} 관측 변화는 재생 중인 안내를 취소하지 않는다`, () => {
    const h = harness(); h.policy.start("A"); h.stable("red", 0);
    const before = h.cancellations.length;
    if (kind === "unknown") h.frame("unknown", 600);
    if (kind === "unselected") h.frame("red", 600, 1, { unselected: true });
    if (kind === "target") h.frame("red", 600, 2);
    if (kind === "error") h.policy.interrupt();
    if (kind === "frame-gap") h.frame("red", 600, 1, { frameId: 10 });
    assert.equal(h.cancellations.length, before);
  });
}

test("감지 공백 이후 사용자가 횡단 중일 수 있으면 대기를 지시하지 않는다", () => {
  const h = harness(); h.policy.start("A");
  h.stable("red", 0); h.stable("green", 600);
  const count = h.speech.length;
  h.frame("unknown", 1200); h.stable("green", 1400);
  assert.equal(h.speech.length, count);
  assert.equal(h.updates.at(-1).text, "초록불입니다.");
});

test("음성 안내 시작 이전 캡처·이전 세션·중복 응답·만료 응답을 안내 근거로 쓰지 않는다", () => {
  const h = harness(); h.setTime(1000); h.policy.start("A");
  h.frame("red", 500); h.frame("red", 1000, 1, { session: "old" });
  h.frame("red", 1100, 1, { frameId: 3 });
  h.frame("red", 1300, 1, { frameId: 3 });
  h.frame("red", 1500, 1, { frameId: 3 });
  h.frame("green", 1700, 1, { receivedAt: 4000 });
  assert.equal(h.speech.length, 1);
});

test("음성 종료 뒤 늦은 응답과 타이머가 재생을 재개하지 않는다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0);
  h.policy.stop(); h.stable("green", 600); h.tick(10000);
  assert.equal(h.speech.length, 2);
});

test("새 안내는 이전 초록 전환 이력을 재사용하지 않고 모의 음성을 구분한다", () => {
  const h = harness(); h.policy.start("A"); h.stable("red", 0); h.stable("green", 600);
  h.setTime(1200); h.policy.start("B", true);
  for (const at of [1200, 1400, 1600]) h.frame("green", at, 1, { session: "B" });
  assert.equal(h.messages().at(-1), "모의 신호. 초록불입니다. 다음 초록 신호를 기다려 주세요.");
});

test("도보 주의 단계는 말하지 않고 위험 단계만 범주에 맞게 안내한다", () => {
  const h = harness(); h.policy.start("A", false, "walking");
  const accept=(frameId,at,event) => {
    h.setTime(at);
    h.policy.accept({session_id:"A",frame_id:frameId,event},at);
  };
  accept(1,0,{type:"walking_warning",level:"caution",voice_category:null,voice_event_id:null});
  accept(2,200,{type:"walking_warning",level:"danger",voice_category:"person",voice_event_id:11});
  accept(3,400,{type:"walking_warning",level:"danger",voice_category:"vehicle",voice_event_id:12});
  accept(4,600,{type:"walking_warning",level:"danger",voice_category:"obstacle",voice_event_id:13});
  assert.deepEqual(h.messages(), ["장애물 안내를 시작합니다.","위험! 사람이 있음.",
    "위험! 차량이 있음.","위험! 장애물이 있음."]);
});

test("같은 도보 위험 객체는 5초 뒤에만 반복하고 새 객체는 즉시 안내한다", () => {
  const h = harness(); h.policy.start("A", false, "walking");
  const danger=(frameId,at,eventId) => {
    h.setTime(at);
    h.policy.accept({session_id:"A",frame_id:frameId,
      event:{type:"walking_warning",level:"danger",voice_category:"person",voice_event_id:eventId}},at);
  };
  danger(1,0,7); danger(2,1000,7); danger(3,4999,7); danger(4,5000,7); danger(5,5200,8);
  assert.deepEqual(h.messages(), ["장애물 안내를 시작합니다.","위험! 사람이 있음.",
    "위험! 사람이 있음.","위험! 사람이 있음."]);
});

test("위험 장애물 음원만 5배속으로 재생한다", () => {
  const h = ttsHarness();
  h.player.speak("위험! 사람이 있음.");
  assert.equal(h.plays[0].playbackRate,5);
  h.plays[0].start(); h.plays[0].end();
  h.player.speak("빨간불입니다.");
  assert.equal(h.plays[1].playbackRate,1);
});

function ttsHarness({ supported = true } = {}) {
  const media = require("./helpers/audio.cjs").audioHarness();
  const errors = [], statuses = [], timers = new Map();
  let now = 0, next = 0;
  const context = {
    window: supported ? { Audio: media.Audio } : {},
    setTimeout(fn) { timers.set(++next, fn); return next; }, clearTimeout: id => timers.delete(id),
  };
  vm.runInNewContext(fs.readFileSync("backend/static/js/tts.js", "utf8"), context);
  return { player: context.window.GTts.create({ onError: e => errors.push(e), onStatus: s => statuses.push(s), now: () => now }),
    ...media, errors, statuses, timers, setTime: t => { now = t; } };
}

test("같은 재생기에서 앞선 안내가 끝난 뒤 요청 순서대로 이어 읽는다", async () => {
  const h = ttsHarness();
  h.player.speak("초록불로 바뀌었습니다.", 1500);
  const first = h.plays[0]; first.start();
  h.player.speak("빨간불입니다.", 1700);
  h.player.speak("신호를 확인할 수 없습니다.", 1800);
  assert.equal(h.plays.length, 1);
  assert.equal(h.pauses.length, 0);
  // 대기 중 원래 유효 시각을 지나도 순서를 지켜 재생한다.
  h.setTime(5000); first.end();
  assert.equal(h.plays[1].src, "/static/audio/ko-v1/red.mp3");
  first.start(); first.end(); first.error(2);
  await Promise.resolve();
  assert.equal(h.plays.length, 2);
  h.plays[1].start(); h.setTime(7000); h.plays[1].end();
  assert.equal(h.plays[2].src, "/static/audio/ko-v1/missing.mp3");
  h.plays[2].start(); h.plays[2].end();
  assert.equal(h.instances.length, 1);
  assert.equal(h.pauses.length, 0);
  assert.equal(h.errors.length, 0);
  assert.equal(h.timers.size, 0);
});

test("로딩 중 들어온 후속 안내도 현재 재생 요청을 취소하지 않는다", () => {
  const h = ttsHarness();
  h.player.speak("빨간불입니다.");
  h.player.speak("초록불로 바뀌었습니다.");
  assert.equal(h.plays.length, 1);
  assert.equal(h.pauses.length, 0);
  h.plays[0].start(); h.plays[0].end();
  assert.equal(h.plays[1].src, "/static/audio/ko-v1/green-changed.mp3");
});

test("끄기는 현재 음성과 대기열을 모두 비우고 늦은 종료로 재개하지 않는다", async () => {
  const h = ttsHarness();
  h.player.speak("빨간불입니다."); h.plays[0].start();
  h.player.speak("초록불로 바뀌었습니다.");
  h.player.cancel();
  h.plays[0].end(); h.plays[0].error(2);
  await Promise.resolve();
  assert.equal(h.plays.length, 1);
  assert.equal(h.pauses.length, 1);
  assert.equal(h.timers.size, 0);
  h.player.speak("신호 안내를 시작합니다.");
  h.plays[1].start(); h.plays[1].end();
  assert.equal(h.plays.length, 2);
  assert.equal(h.errors.length, 0);
});

test("재생 오류가 나면 남은 대기열도 비우고 오류를 알린다", () => {
  const h = ttsHarness();
  h.player.speak("빨간불입니다."); h.plays[0].start();
  h.player.speak("초록불로 바뀌었습니다.");
  h.plays[0].error(2); h.plays[0].end();
  assert.equal(h.errors.length, 1);
  assert.equal(h.plays.length, 1);
  assert.equal(h.timers.size, 0);
});

test("종료 콜백에서 추가된 안내는 이미 대기 중인 안내 뒤에 재생한다", () => {
  const h = ttsHarness();
  h.player.speak("빨간불입니다.", 8000, { onEnd: () => h.player.speak("신호를 확인할 수 없습니다.") });
  h.player.speak("초록불로 바뀌었습니다.");
  h.plays[0].start(); h.plays[0].end();
  assert.equal(h.plays[1].src, "/static/audio/ko-v1/green-changed.mp3");
  h.plays[1].start(); h.plays[1].end();
  assert.equal(h.plays[2].src, "/static/audio/ko-v1/missing.mp3");
});

test("일반·모의·음성 확인 문구 전부 실제 포함된 MP3에 대응한다", () => {
  const h = ttsHarness();
  const root = "backend/static/audio/ko-v1/";
  const { clips } = JSON.parse(fs.readFileSync(root + "manifest.json", "utf8"));
  assert.equal(Object.keys(clips).length, 19);
  assert.deepEqual(fs.readdirSync(root).filter(name => name.endsWith(".mp3")).sort(),
    Object.keys(clips).sort());
  for (const [filename, text] of Object.entries(clips)) {
    assert.equal(h.player.speak(text), true);
    const version = filename.startsWith("danger-") ? "?v=walking-audio-v2" : "";
    assert.equal(h.plays.at(-1).src, "/static/audio/ko-v1/" + filename + version);
    h.plays.at(-1).start(); h.plays.at(-1).end();
    const bytes = fs.readFileSync(root + filename);
    assert.ok(bytes.length > 1000);
    assert.ok(bytes.subarray(0, 3).toString() === "ID3" || (bytes[0] === 255 && (bytes[1] & 224) === 224));
  }
});

test("미지원 브라우저와 등록되지 않은 문구는 재생 없이 오류를 알린다", () => {
  for (const supported of [true, false]) {
    const h = ttsHarness({ supported });
    assert.equal(h.player.speak("등록되지 않은 문구"), false);
    assert.equal(h.plays.length, 0);
    assert.equal(h.errors.length, 1);
    assert.equal(h.timers.size, 0);
  }
});

test("발화 종료 콜백은 실제 재생과 종료 후 한 번만 호출한다", async () => {
  const h = ttsHarness(); let ended = 0;
  h.player.speak("신호 안내를 시작합니다.", 8000, { onEnd: () => ended++ });
  h.plays[0].end();
  assert.equal(ended, 0);
  h.plays[0].start();
  await Promise.resolve();
  assert.equal(ended, 0);
  h.plays[0].end(); h.plays[0].end();
  assert.equal(ended, 1);
  assert.equal(h.timers.size, 0);
});

test("play Promise가 확인한 재생 성공도 상태에 표시한다", async () => {
  const h = ttsHarness(); h.player.speak("신호 안내를 시작합니다.");
  h.plays[0].resolve();
  await Promise.resolve();
  assert.match(h.statuses.at(-1), /재생 중/);
  const timer = [...h.timers.keys()][0];
  h.plays[0].start();
  assert.equal([...h.timers.keys()][0], timer);
});

test("재생 완료가 오지 않으면 타이머로 종료해 시작 준비가 무한정 멈추지 않는다", () => {
  const h = ttsHarness(); h.player.speak("신호 안내를 시작합니다."); h.plays[0].start();
  [...h.timers.values()][0]();
  assert.match(h.errors[0], /끝나지 않아/);
  assert.equal(h.instances[0].src, "");
});

test("브라우저 재생 차단과 미지원 음원을 구분한다", async () => {
  for (const [name, expected] of [["NotAllowedError", /사이트 소리 허용/], ["NotSupportedError", /음원을 불러오거나/]]) {
    const h = ttsHarness(); h.player.speak("신호 안내를 시작합니다.");
    h.plays[0].reject({ name });
    await Promise.resolve();
    assert.match(h.errors[0], expected);
    assert.equal(h.timers.size, 0);
  }
});

test("미디어의 서버 연결 오류와 디코딩 오류를 구분한다", () => {
  for (const [code, expected] of [[2, /서버 연결/], [3, /파일을 재생/], [4, /파일을 재생/]]) {
    const h = ttsHarness(); h.player.speak("신호 안내를 시작합니다.");
    h.plays[0].error(code);
    assert.match(h.errors[0], expected);
    assert.equal(h.timers.size, 0);
  }
});

test("재생 시작이 만료 시각 이후면 취소하고 오류를 알린다", async () => {
  for (const trigger of ["start", "resolve"]) {
    const h = ttsHarness(); h.player.speak("초록불로 바뀌었습니다.", 1500);
    h.setTime(1501); h.plays[0][trigger]();
    await Promise.resolve();
    assert.equal(h.errors.length, 1);
    assert.equal(h.timers.size, 0);
    assert.equal(h.instances[0].src, "");
  }
});

test("요청 시점에 만료된 신호는 재생하지 않는다", () => {
  const h = ttsHarness(); h.setTime(1500);
  assert.equal(h.player.speak("초록불로 바뀌었습니다.", 1500), false);
  assert.equal(h.plays.length, 0);
});

test("재생 이벤트가 오지 않아도 타이머가 로딩 중인 음원을 취소한다", () => {
  const h = ttsHarness(); h.player.speak("초록불로 바뀌었습니다.", 1500);
  [...h.timers.values()][0]();
  assert.equal(h.errors.length, 1);
  assert.equal(h.timers.size, 0);
  assert.equal(h.instances[0].src, "");
});

test("종료 후 도착한 이전 재생 오류와 성공은 안내를 재개하지 않는다", async () => {
  for (const trigger of ["reject", "resolve"]) {
    const h = ttsHarness(); h.player.speak("초록불로 바뀌었습니다.", 1500);
    h.player.cancel();
    h.plays[0][trigger]({ name: "AbortError" }); h.plays[0].error(2);
    await Promise.resolve();
    assert.equal(h.errors.length, 0);
    assert.equal(h.timers.size, 0);
    assert.match(h.statuses.at(-1), /취소/);
  }
});

test("동기 재생 오류도 호출자에게 알리고 타이머를 정리한다", () => {
  const h = ttsHarness(); h.instances[0].play = () => { throw new Error("device failure"); };
  assert.equal(h.player.speak("초록불로 바뀌었습니다.", 1500), false);
  assert.equal(h.errors.length, 1);
  assert.equal(h.timers.size, 0);
});
