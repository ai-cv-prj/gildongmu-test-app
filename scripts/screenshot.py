"""가짜 카메라로 UI 화면을 캡처한다. 서버가 127.0.0.1:8000 에 떠 있어야 한다.
사용: .venv/bin/python scripts/screenshot.py [base_url]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None, args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
    ctx = browser.new_context(
        viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True,
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        permissions=["camera"],
    )
    page = ctx.new_page()
    font = os.environ.get("FONT_PATH")  # 서버 PC에 한글 폰트가 없을 때 캡처용으로만 주입
    if font:
        page.route("**/__shot_font.ttf", lambda r: r.fulfill(path=font, content_type="font/ttf"))
    page.on("console", lambda m: print("[console]", m.type, m.text) if m.type in ("error", "warning") else None)
    page.goto(BASE)
    page.wait_for_selector("#model-select option", state="attached")
    if font:
        page.add_style_tag(content="@font-face{font-family:ShotKR;src:url(/__shot_font.ttf);font-weight:100 900}"
                           "html,body,button,select,input{font-family:ShotKR,sans-serif!important}")
        page.evaluate("document.fonts.load('16px ShotKR').then(() => document.fonts.ready)")
        time.sleep(0.5)
    page.screenshot(path=str(OUT / "01_idle.png"))
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(0.3)
    page.screenshot(path=str(OUT / "01b_idle_settings.png"))
    page.evaluate("window.scrollTo(0, 0)")

    page.click("#btn-camera")
    page.wait_for_function("document.getElementById('camera-frame').classList.contains('live')")
    time.sleep(0.5)
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.3)
    page.screenshot(path=str(OUT / "02_camera.png"))

    page.click("#btn-start")
    page.wait_for_function("parseInt(document.getElementById('m-frames').textContent) >= 15", timeout=20000)
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.3)
    page.screenshot(path=str(OUT / "03_running_traffic.png"))

    page.click("#btn-stop")
    page.wait_for_selector("#summary:not([hidden])")
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(0.3)
    page.screenshot(path=str(OUT / "04_stopped.png"))
    page.evaluate("window.scrollTo(0, 0)")

    # 도보 장애물 모드 + 직접 입력 기기
    page.click("#mode-seg button[data-mode=walking]")
    page.select_option("#device-select", "__custom__")
    page.fill("#device-custom", "Pixel 8")
    page.click("#btn-start")
    page.wait_for_selector("#session-badge[data-state=running]")
    page.wait_for_function("parseInt(document.getElementById('m-frames').textContent) < 5")
    page.wait_for_function("parseInt(document.getElementById('m-frames').textContent) >= 12", timeout=20000)
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.3)
    page.screenshot(path=str(OUT / "05_running_walking.png"))
    page.click("#btn-stop")
    page.wait_for_selector("#summary:not([hidden])")
    browser.close()
print("saved to", OUT)
