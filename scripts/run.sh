#!/usr/bin/env bash
# 서버 실행. 처음이면 venv 를 만들고 의존성을 설치한다.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "[run] .venv 생성 중 ($PY)"
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r backend/requirements.txt
fi
[ -f .env ] || { cp .env.example .env; echo "[run] .env 를 만들었습니다"; }
set -a; source .env; set +a
HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8000}"

# 포트가 이미 쓰이고 있으면 uvicorn 의 긴 오류 대신 바로 알려준다
if .venv/bin/python -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT))==0 else 1)"; then
  echo "[run] 포트 $PORT 를 이미 다른 프로그램이 쓰고 있습니다."
  echo "      - 이전에 켠 서버가 남아 있으면 그 터미널에서 Ctrl+C 로 끄세요."
  echo "      - 아니면 .env 의 APP_PORT 를 8001 같은 다른 번호로 바꾸세요. (tunnel.sh 도 같은 값을 읽습니다)"
  exit 1
fi

echo "[run] http://127.0.0.1:$PORT  (휴대폰 접속은 다른 터미널에서 ./scripts/tunnel.sh)"
exec .venv/bin/python -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT" --workers 1
