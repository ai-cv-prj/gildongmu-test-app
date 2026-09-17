#!/usr/bin/env bash
# 휴대폰에서 접속할 HTTPS 임시 주소를 만든다. 서버(run.sh)가 먼저 켜져 있어야 한다.
# 출력되는 https://xxxx.trycloudflare.com 주소를 "본인" 휴대폰 브라우저에서 연다.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }
PORT="${APP_PORT:-8000}"
command -v cloudflared >/dev/null || { echo "[tunnel] cloudflared 가 없습니다. README 의 설치 안내를 보세요."; exit 1; }
curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null || { echo "[tunnel] 포트 $PORT 에 서버가 없습니다. 먼저 ./scripts/run.sh 를 실행하세요."; exit 1; }
exec cloudflared tunnel --url "http://127.0.0.1:$PORT"
