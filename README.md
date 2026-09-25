# gildongmu-test-app

Gildongmu 모델을 **휴대폰 카메라로 실시간 테스트**하는 팀 내부용 앱입니다.

- 휴대폰 브라우저에서 후면 카메라를 켜면 프레임이 실시간으로 **본인 PC**로 전송됩니다. 촬영 후 업로드하는 방식이 아닙니다.
- 본인 PC가 선택한 기능(신호등 / 도보 장애물 / 버스)의 모델로 추론하고, 휴대폰 화면에 박스와 처리 시간을 보여줍니다.
- 테스트 시작부터 종료까지 전송된 모든 프레임과 추론 결과가 **본인 PC의 폴더에 자동 저장**됩니다.
- 모델이 없어도 Mock(가짜 박스) 모델로 전체 흐름을 먼저 확인할 수 있습니다.

신호등 검출·선택·추적·색상 분류와 보행 영역 추론 코드가 포함되어 있습니다. **모델 가중치는 각자 본인 PC에 넣습니다.** 버스 추론은 기능별 파이프라인에 구현합니다.

```text
 본인 휴대폰                         본인 PC (GPU)
┌───────────────┐   HTTPS 터널    ┌─────────────────────────────────┐
│ 브라우저       │ ─────────────▶ │ FastAPI 서버 (기본 포트 8000)    │
│ 카메라·박스·지표│ ◀───────────── │  ├ 추론: backend/app/inference/  │
└───────────────┘    결과 JSON    │  ├ 가중치: backend/models/       │
                                  │  └ 기록: backend/data/sessions/  │
                                  └─────────────────────────────────┘
```

## 목차

1. [준비물](#1-준비물)
2. [처음 설치](#2-처음-설치)
3. [실행하고 휴대폰으로 접속하기](#3-실행하고-휴대폰으로-접속하기)
4. [포트와 네트워크 주의사항 (4명이 각자 쓸 때)](#4-포트와-네트워크-주의사항-4명이-각자-쓸-때)
5. [본인 모델 연결하기](#5-본인-모델-연결하기)
6. [기록은 어디에 어떻게 저장되나](#6-기록은-어디에-어떻게-저장되나)
7. [Git 사용 규칙](#7-git-사용-규칙)
8. [문제 해결](#8-문제-해결)
9. [참고: 설정, 코드 구조, 테스트](#9-참고)

---

## 1. 준비물

| 항목 | 설명 |
| --- | --- |
| Python 3.12 | `python3 --version` 으로 확인. 보행 위험 기능의 기준 버전입니다. |
| Git | 코드 받기용 |
| cloudflared | 휴대폰 접속용 HTTPS 주소를 만들어 줍니다. 계정과 로그인은 필요 없습니다. |
| 본인 모델 가중치 | `.pt` `.pth` `.onnx` `.engine` 파일. 없으면 Mock 으로 먼저 실행해 볼 수 있습니다. |
| 휴대폰 | iPhone 은 Safari, Android 는 Chrome 을 권장합니다. |

cloudflared 설치:

```bash
# Ubuntu / WSL
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/

# macOS
brew install cloudflared

# Windows (PowerShell)
winget install --id Cloudflare.cloudflared
```

앱 실행에는 Node.js 가 필요 없습니다. 화면은 빌드가 없는 HTML/JS 이고 서버가 함께 제공합니다.

## 2. 처음 설치

### Linux / WSL / macOS

```bash
git clone https://github.com/ai-cv-prj/gildongmu-test-app.git
cd gildongmu-test-app
./scripts/run.sh
```

`run.sh` 가 처음 한 번 `.venv` 생성, 패키지 설치, `.env` 생성을 자동으로 하고 서버를 켭니다.

- `python3 -m venv` 가 실패하면 `sudo apt install python3-venv` 를 먼저 실행하세요.
- Python 3.12를 지정하려면 `PYTHON=python3.12 ./scripts/run.sh` 로 실행합니다. 기존 `.venv`가 있으면 먼저 그 환경의 버전을 확인하세요.

### Windows (PowerShell, WSL 을 쓰지 않는 경우)

```powershell
git clone https://github.com/ai-cv-prj/gildongmu-test-app.git
cd gildongmu-test-app
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Windows 에서는 `scripts/*.sh` 를 쓸 수 없으므로 위 마지막 줄이 서버 실행 명령입니다. 이 문서의 `.venv/bin/python` 은 Windows 에서 `.venv\Scripts\python` 으로 바꿔 읽으면 됩니다. 이 Windows 절차는 개발 PC(WSL)에서 직접 실행해 보지는 못했습니다. 막히면 알려주세요.

### 설치 확인

PC 브라우저에서 http://127.0.0.1:8000 을 엽니다. 화면이 뜨고 오른쪽 위 "서버", "저장" 표시가 초록색이면 정상입니다. PC 에 웹캠이 있으면 이 주소에서도 카메라 시작과 Mock 테스트를 해볼 수 있습니다.

## 3. 실행하고 휴대폰으로 접속하기

터미널 두 개를 씁니다. 테스트하는 동안 **둘 다 켜 둬야** 합니다.

```bash
# 터미널 1: 서버
./scripts/run.sh

# 터미널 2: 휴대폰용 HTTPS 주소 만들기
./scripts/tunnel.sh
```

Windows PowerShell 에서는 터미널 2 에서 `cloudflared tunnel --url http://127.0.0.1:8000` 을 실행합니다.

터미널 2 에 아래처럼 주소가 나옵니다. 이 주소를 **본인 휴대폰**에서 엽니다. LTE/5G 여도 되고, PC 와 다른 네트워크여도 됩니다.

```text
Your quick Tunnel has been created! Visit it at:
https://random-words-here.trycloudflare.com
```

> **왜 터널이 필요한가요?** 브라우저는 HTTPS 또는 localhost 에서만 카메라를 열어 줍니다. `http://192.168.x.x:8000` 같은 주소로 접속하면 화면은 떠도 카메라가 차단됩니다.

### 테스트 순서

1. **기능**을 고릅니다: 신호등 / 도보 장애물 / 버스.
2. **모델**을 고릅니다. 신호등은 `best_YOLO_v2.pt`가 있으면 기본으로 선택됩니다. v2가 없거나 다른 기능이면 사용 가능한 첫 번째 실제 모델을 선택하고, 실제 모델이 없으면 Mock을 선택합니다. 목록에서 다른 모델로 바꿀 수 있습니다.
3. **촬영 기기**를 고릅니다. 목록에 없으면 "직접 입력…"을 골라 적습니다. 마지막 선택은 휴대폰이 기억합니다.
4. 필요하면 **메모**를 적습니다. 예: `역광`, `야간`, `횡단보도 앞 5m`. 나중에 기록을 구분할 때 유용합니다.
5. **카메라 시작** → 권한 허용 → **테스트 시작**. 이때부터 전송과 저장이 시작됩니다.
6. **테스트 종료**를 누르면 저장된 프레임 수와 폴더 위치가 화면에 나옵니다.

테스트 중에는 기능과 모델을 바꿀 수 없습니다. 바꾸려면 종료한 뒤 다시 시작합니다. 화면을 끄거나 다른 앱으로 넘어가면 전송이 일시정지되고, 돌아오면 이어집니다.

신호등 모드에서는 선택된 신호등에 `안내 대상`을 표시하고 한국어 음성으로 안내합니다. 소리는 테스트 전·중·후에 켜거나 끌 수 있으며, 테스트 전 `음성 확인`도 가능합니다. 선택·추적 기준과 화면 표시는 [신호등 모듈 문서](docs/traffic-signal.md)를 참고하세요.

도보 장애물의 위험 판단 기준·설정·결과 영상은 [도보 위험 판단 문서](docs/walking-risk.md)에 정리돼 있습니다.

### 화면에 나오는 숫자

| 항목 | 의미 |
| --- | --- |
| 추론 | 모델 `infer()` 에 걸린 시간 |
| 서버 처리 | 서버의 디코딩 + 추론 + 원본 이미지 저장 시간. 결과 로그·요약 저장, 응답 전송은 제외 |
| 왕복 | 휴대폰의 캡처 시작부터 결과 수신·JSON 해석까지. JPEG 생성, 서버 처리, 통신이 포함되며 마스크 그리기는 제외 |
| 수신 FPS | 최근 3초 동안 초당 받은 결과 수. 전송 상한은 신호등 5, 나머지 기능 10이며 실제 속도는 처리·통신 시간에 따라 달라집니다. |
| 전송 프레임 / 실패 | 이번 테스트에서 성공한 수와 실패한 수 |

끝낼 때는 두 터미널에서 각각 `Ctrl+C` 를 누릅니다. **터널은 테스트가 끝나면 꼭 끄세요** (4장 참고).

## 4. 포트와 네트워크 주의사항 (4명이 각자 쓸 때)

각자 본인 PC 에서 실행하므로 **팀원끼리 포트가 겹치는 일은 없습니다.** 네 명 모두 기본값 8000 을 그대로 써도 됩니다. 포트는 "그 PC 안에서만" 유일하면 됩니다. 주의할 상황은 아래입니다.

### 4-1. 내 PC 에서 8000 을 이미 다른 프로그램이 쓰는 경우

Jupyter, 다른 FastAPI/Django 서버, 이전에 켜 둔 이 앱 등이 8000 을 잡고 있으면 `run.sh` 가 이렇게 알려줍니다.

```text
[run] 포트 8000 를 이미 다른 프로그램이 쓰고 있습니다.
```

- 이전에 켠 서버가 남아 있는 것이면 그 터미널에서 `Ctrl+C` 로 끕니다.
- 누가 쓰는지 확인: `lsof -i :8000` (Linux/macOS), `netstat -ano | findstr :8000` (Windows)
- 다른 프로그램을 끌 수 없으면 `.env` 의 `APP_PORT` 를 바꿉니다. `tunnel.sh` 도 같은 값을 읽으므로 한 군데만 고치면 됩니다.

```dotenv
APP_PORT=8001
```

Windows 에서 직접 명령을 칠 때는 `--port 8001` 과 `cloudflared tunnel --url http://127.0.0.1:8001` 두 곳을 같이 바꿔야 합니다. **서버 포트와 터널 포트가 다르면 휴대폰에서 502 오류가 납니다.**

### 4-2. 한 대의 PC(연구실 GPU 서버 등)를 여러 명이 같이 쓰는 경우

이때만 포트가 실제로 겹칩니다. 사람마다 저장소를 따로 clone 하고 `.env`의 `APP_PORT`를 8001, 8002처럼 다르게 지정하세요. 기록 폴더도 분리됩니다. GPU 한 장을 공유한다면 **속도를 재는 테스트는 한 번에 한 명만** 하세요.

### 4-3. 터널 주소는 사람마다 다르고, 공유하면 안 됩니다

- `tunnel.sh` 를 실행할 때마다 **새로운 임의 주소**가 나옵니다. 어제 주소는 오늘 안 됩니다. 매번 새 주소를 휴대폰에서 여세요.
- 주소는 **그 주소를 만든 사람의 PC** 로 연결됩니다. 팀원이 단톡방에 올린 주소를 내 휴대폰으로 열면 내 영상이 **그 팀원 PC 에서 추론되고 그 팀원 PC 에 저장**됩니다. 본인 터미널에 나온 주소만 쓰세요.
- 이 앱에는 **로그인이 없습니다.** 주소를 아는 사람은 누구나 접속해 카메라 테스트를 돌리고 저장된 프레임을 API 로 볼 수 있습니다. 주소를 공개된 곳에 올리지 말고, 테스트가 끝나면 터널을 끄세요.
- 한 서버는 **동시에 한 세션만** 받습니다. 누군가 같은 주소로 테스트 중이면 두 번째 사람은 "이미 실행 중인 세션이 있습니다" 오류를 받습니다.

### 4-4. APP_HOST 는 127.0.0.1 그대로 두세요

기본값 `127.0.0.1` 은 "내 PC 안에서만 접속 가능"이라는 뜻입니다. 휴대폰은 터널을 통해 들어오므로 이것으로 충분하고, 방화벽이나 공유기 포트포워딩 설정도 필요 없습니다. `0.0.0.0` 으로 바꾸면 같은 와이파이(학교, 카페)의 누구나 로그인 없이 접속할 수 있게 되므로 바꾸지 마세요.

### 4-5. WSL 사용자

WSL 안에서 서버를 켜도 Windows 브라우저에서 http://127.0.0.1:8000 으로 열립니다. `cloudflared` 도 WSL 안에 설치해서 WSL 터미널에서 실행하면 됩니다. WSL 과 Windows 양쪽에서 동시에 서버를 켜면 같은 8000 을 두고 충돌하니 한쪽에서만 실행하세요.

### 4-6. 테스트 중 PC

노트북이 절전에 들어가면 서버와 터널이 멈춥니다. 야외 테스트 전에 절전을 끄고 전원을 연결해 두세요. PC 의 인터넷이 끊겨도 터널이 끊깁니다.

## 5. 본인 모델 연결하기

**본인 기능의 파이프라인과 관련 모듈**을 수정하고, 가중치는 해당 폴더에 넣습니다. 신호등과 도보 장애물은 기능별 패키지로 나뉘며, 모델 연결의 시작점은 아래 파일입니다.

| 기능 | 수정할 파일 | 가중치 넣는 폴더 |
| --- | --- | --- |
| 신호등 | `backend/app/inference/traffic/pipeline.py` | `backend/models/traffic/` |
| 도보 장애물 | `backend/app/inference/walking/pipeline.py` | `backend/models/walking/` |
| 버스 | `backend/app/inference/bus.py` | `backend/models/bus/` |

### 5-1. 모델 패키지 설치

모델 패키지는 서버와 **같은 가상환경**(`.venv`)에 설치해야 합니다. `requirements.txt`에 공통 버전이 지정되어 있고, `run.sh`가 첫 실행 때 함께 설치합니다. GPU용 PyTorch 빌드는 본인 CUDA 환경에 맞게 설치하세요.

```bash
# GPU 확인. True 가 나와야 GPU 로 추론합니다.
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` 가 나오면 PyTorch가 GPU를 사용하지 못하는 상태입니다. https://pytorch.org/get-started/locally/ 에서 본인 GPU 와 CUDA 에 맞는 설치 명령을 확인하세요. RTX 50 시리즈는 CUDA 12.8 이상 빌드가 필요합니다. 이 저장소의 `requirements.txt`에는 `torch`, `torchvision`, `ultralytics` 버전이 지정되어 있습니다.

### 5-2. 가중치 파일 넣기

**`git clone`이나 `git pull`로는 모델 가중치가 다운로드되지 않습니다.** `backend/models/`의 가중치는 Git 관리 대상에서 제외되어 있습니다. 저장소에 가중치 다운로드 링크는 없으므로, 모델 공유자에게 드라이브 등의 공유 링크나 파일을 별도로 받아야 합니다.

현재 신호등 모델을 실행하려면 아래 **두 파일을 모두** 받아 저장소 루트 기준으로 배치하세요. `classifier/` 폴더가 없으면 직접 만듭니다.

```text
backend/models/traffic/
├── best_YOLO_v2.pt              # 신호등·횡단보도 검출
└── classifier/
    └── best_MobileNet.pt        # 신호등 색상 분류
```

예를 들어 두 파일을 `~/Downloads/`에 다운로드했다면 WSL/Linux에서 다음과 같이 복사합니다.

```bash
mkdir -p backend/models/traffic/classifier
cp ~/Downloads/best_YOLO_v2.pt backend/models/traffic/
cp ~/Downloads/best_MobileNet.pt backend/models/traffic/classifier/
```

Windows에서도 같은 폴더 구조에 넣습니다. `best_MobileNet.pt`가 없으면 YOLO가 목록에 보여도 테스트 시작 시 로딩에 실패합니다. 분류기는 별도 선택 항목이 아닙니다.

- 신호등 파이프라인은 YOLO `.pt` 가중치를 사용하며, `best_YOLO_v2.pt`가 있으면 기본으로 선택됩니다. 모델 목록은 휴대폰에서 새로고침하면 갱신됩니다.
- `.pt` `.pth` `.onnx` `.engine` 파일은 목록에 표시되지만, 실제 로딩은 해당 파이프라인이 지원하는 형식이어야 합니다.
- **가중치를 바꿔 비교하려면 파일을 여러 개 넣으세요.** `best_v1.pt`, `best_v2_aug.pt` 처럼 이름으로 구분하면 각각 선택지가 되고, 어떤 파일로 찍은 기록인지 세션 요약에 파일 이름과 해시가 남습니다.
- 인식됐는지 확인:

```bash
.venv/bin/python scripts/check_model.py --list
```

목록에서 `best_YOLO_v2.pt`가 인식되는지 확인하세요. 목록 조회는 분류기 로딩까지 검사하지 않으므로, 두 파일을 배치한 뒤 앱에서 신호등 모델을 선택하고 테스트를 시작해 로딩도 확인합니다.

### 5-3. 추론 코드 넣기

본인 기능의 파일을 열어 세 군데를 채웁니다. 아래는 ultralytics YOLO 로 채운 `traffic/pipeline.py` 의 완성 예시입니다. 이 예시는 표준 ultralytics API 로 작성했지만, 개발 PC 에 ultralytics 를 설치해 실행해 보지는 않았습니다. 5-4 의 점검 스크립트로 본인 환경에서 확인하세요.

```python
CLASS_NAMES: dict[int, str] = {0: "red_light", 1: "green_light"}   # ① 본인 data.yaml 의 클래스 순서


class TrafficPipeline:
    mode = "traffic"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.weights = spec.weights      # 화면에서 선택한 가중치 파일 경로
        self.model = None

    def load(self) -> None:              # ② 테스트 시작 시 한 번만 호출됩니다
        from ultralytics import YOLO
        self.model = YOLO(str(self.weights))

    def reset_session(self, session_id: str) -> None: ...
    def close_session(self, session_id: str) -> None: ...

    def infer(self, frame_bgr, context):  # ③ 프레임마다 호출됩니다
        h, w = frame_bgr.shape[:2]
        result = self.model.predict(frame_bgr, conf=context.confidence, verbose=False)[0]

        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cid = int(b.cls[0])
            detections.append({
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, str(cid)),
                "confidence": float(b.conf[0]),
                "box": normalize_box(x1, y1, x2, y2, w, h),   # 픽셀 → 0~1 좌표
                "track_id": None,
                "extra": {},
            })

        # 신호 상태 판정: 가장 신뢰도 높은 검출 기준 (본인 로직으로 교체)
        state = "unknown"
        if detections:
            top = max(detections, key=lambda d: d["confidence"])
            state = {"red_light": "red", "green_light": "green"}.get(top["class_name"], "unknown")
        return {"detections": detections, "event": {"type": "traffic_signal", "signal_state": state}}
```

지켜야 할 규칙:

| 규칙 | 설명 |
| --- | --- |
| 입력은 **OpenCV BGR** `numpy` 배열 | 모델이 RGB 를 요구하면 `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)` 로 직접 변환하세요. ultralytics 는 BGR 배열을 그대로 받습니다. |
| 입력 크기는 매번 다를 수 있음 | 기본 전송 크기는 긴 변 최대 신호등 960px, 나머지 기능 640px 이고 세로 영상도 옵니다. `h, w` 를 매 프레임 읽으세요. |
| 좌표는 **0~1 정규화 xyxy** | 픽셀 좌표는 `normalize_box(x1, y1, x2, y2, w, h)` 로 변환합니다. xywh 나 중심 좌표 형식이면 먼저 xyxy 로 바꾸세요. |
| 값은 **파이썬 기본 타입** | tensor 나 numpy 값은 `float()`, `int()` 로 변환합니다. 안 하면 저장 단계에서 오류가 납니다. |
| `load()` 에서만 모델 로딩 | `infer()` 안에서 매번 로딩하면 프레임마다 몇 초씩 걸립니다. |
| 신뢰도 기준은 `context.confidence` | 기본값은 신호등 0.25, 실제 보행 모델은 `config/walking_risk.yaml`의 `yolo.conf`, 보행 Mock과 버스는 0.4입니다. |

기능별 `event` 형식:

```python
# traffic : signal_state 는 "red" / "green" / "unknown" 중 하나
{"type": "traffic_signal", "signal_state": "green"}

# walking : warning 이 True 면 휴대폰 화면에 주황색 경고 배너가 뜹니다
{"type": "walking_warning", "warning": True, "warning_text": "전방 2m 볼라드"}

# bus : 번호를 아직 못 읽었으면 bus_number 는 None
{"type": "bus_detection", "bus_number": "271", "is_target": True}
```

검출마다 추가 정보를 남기고 싶으면 `"extra": {"distance_m": 2.1}` 처럼 `extra` 에 넣으세요. 기록 파일에 그대로 저장됩니다. 추적 ID 가 있으면 `track_id` 에 정수로 넣으면 박스 라벨에 `#3` 처럼 표시됩니다.

**프레임 사이에 상태가 필요한 경우**(추적기, N프레임 연속 판정 등)에는 `reset_session()` 에서 초기화하세요. 테스트를 시작할 때마다 호출되므로 이전 테스트의 상태가 섞이지 않습니다.

```python
def reset_session(self, session_id):
    self.history = []          # 예: 최근 5프레임의 판정
```

버스처럼 검출 → 추적 → OCR 로 모델이 여러 개 이어지는 경우에도 파일 하나 안에서 `load()` 에 모두 로딩하고 `infer()` 에서 순서대로 호출하면 됩니다. 보조 가중치(OCR 등)는 인식되는 확장자와 겹치지 않게 `backend/models/bus/ocr/` 같은 **하위 폴더**에 넣고 코드에서 경로를 직접 지정하세요. 하위 폴더의 파일은 모델 목록에 뜨지 않습니다.

### 5-4. 휴대폰 없이 먼저 점검하기

이미지 한 장으로 로딩, 추론, 반환 형식을 한 번에 검사합니다. **휴대폰 테스트 전에 꼭 통과시키세요.**

```bash
.venv/bin/python scripts/check_model.py --mode traffic --image 테스트사진.jpg
```

```text
모델: traffic-best  (backend/models/traffic/best.pt)
[통과] load()  1840 ms
[통과] infer() 5회  첫 회 310 ms, 이후 평균 24 ms
[통과] 반환 형식  검출 1개, event={'type': 'traffic_signal', 'signal_state': 'green'}
   - green_light 0.91 {'x1': 0.41, 'y1': 0.12, 'x2': 0.47, 'y2': 0.25}
박스를 그린 이미지: check_output/traffic_테스트사진.jpg
```

- `[실패]` 가 나오면 어느 단계에서 무엇이 틀렸는지 알려줍니다. 예: 좌표가 0~1 을 벗어남, tensor 를 변환하지 않음.
- `check_output/` 에 저장된 이미지를 열어 **박스 위치가 맞는지 눈으로 확인**하세요. 위치가 어긋나면 좌표 형식(xyxy / xywh)이나 정규화가 틀린 것입니다.
- 특정 가중치를 지정하려면 `--model traffic-best-v2` 를 붙입니다. ID 는 `--list` 로 확인합니다.
- "이후 평균" 시간이 100ms 를 넘으면 추론만으로도 10FPS 모드의 전송 상한을 못 따라갑니다. 신호등 모드는 상한 5FPS이므로 200ms와 비교하세요. 실제 속도에는 캡처·저장·통신 시간도 포함됩니다. GPU 사용 여부도 확인하세요.

### 5-5. 휴대폰으로 실제 테스트

1. `./scripts/run.sh` 와 `./scripts/tunnel.sh` 실행
2. 휴대폰에서 기능을 고르면 모델 목록에 본인 가중치 파일이 기본 선택되어 있습니다.
3. 첫 "테스트 시작"은 모델 로딩 때문에 몇 초 걸릴 수 있습니다. 두 번째부터는 바로 시작됩니다.
4. 코드를 고쳤다면 **서버를 재시작**해야 반영됩니다 (`Ctrl+C` 후 `run.sh`). 터널은 그대로 둬도 주소가 유지됩니다.

## 6. 기록은 어디에 어떻게 저장되나

데이터베이스는 없습니다. 기록은 전부 **서버를 켠 PC 의 폴더와 파일**이고, 테스트 한 번이 폴더 하나입니다.

```text
backend/data/sessions/
└─ 20260917_143821_iphone-15-pro_traffic/     날짜_시각_기기_기능 (PC 로컬 시각)
   ├─ manifest.json                           세션 요약
   ├─ frames/00000001.jpg ...                 휴대폰이 보낸 입력 프레임
   ├─ results.jsonl                           프레임당 한 줄의 서버 추론 결과
   ├─ client_timings.jsonl                    프레임당 한 줄의 휴대폰 지연 측정값
   ├─ realtime_overlay.webm                   테스트 종료 시 업로드하는 실시간 탐지 녹화
   └─ annotated/results.mp4                   일반 세션의 탐지 결과 영상
```

- **manifest.json**: 기기, 메모, 기능, 모델 ID, 가중치 파일 이름과 해시, 시작·종료 시각, 상태, 프레임 수, 실패 수, 평균·p95 처리 시간, 전송 설정, 휴대폰 브라우저 정보.
- **frames/**: 신호등은 초당 최대 5장·긴 변 최대 960px, 나머지 기능은 초당 최대 10장·긴 변 최대 640px JPEG. 파일 번호가 프레임 번호입니다. 박스가 그려지지 않은 원본이라 다른 가중치로 다시 추론해 볼 수 있습니다.
- **results.jsonl**: 프레임 번호, 촬영 시각, 서버 수신 시각, 이미지 크기, 검출 목록, event, 단계별 처리 시간, 이미지 경로. 실패한 프레임은 `error` 에 원인이 남습니다.
- **결과 영상**: 일반 세션은 `annotated/results.mp4`, 보행 위험 세션은 `result_visualized.mp4`를 만듭니다. 보행 영상의 촬영 간격 재현과 저장물은 [도보 위험 판단 문서](docs/walking-risk.md#필수-저장물)를 참고하세요.
- **video_status**: 일반 세션의 영상 생성 상태는 세션 조회 API와 `manifest.json`에서 `pending`·`ready`·`failed`·`no_frames`로 확인합니다.
- **client_timings.jsonl**: 아래 표의 휴대폰 측정값. 2.5초마다 최대 25건씩 전송하며, 테스트 종료 시 마지막 기록까지 저장합니다. 새 테스트부터 생성됩니다.
- 상태는 `running` / `completed` / `aborted` 입니다. 서버가 테스트 도중 꺼지면 다음에 켤 때 `aborted` 로 바뀌고, 그때까지 받은 프레임은 그대로 남아 있습니다.
- **폴더가 곧 기록입니다.** 필요 없는 테스트는 폴더를 지우면 되고, 팀에 공유할 때는 폴더를 압축해 보내면 됩니다.
- 용량은 10분 테스트에 대략 200~400MB 입니다. 디스크 여유가 `MIN_FREE_DISK_GB`(기본 2GB)보다 적으면 새 테스트가 시작되지 않습니다.

### 실시간 지연 분석 로그

`results.jsonl`과 `client_timings.jsonl`을 같은 `session_id`, `frame_id`로 연결합니다. 시간 단위는 ms입니다.

| 항목 | 의미 |
| --- | --- |
| `inference_ms`, `server_ms` | 모델 추론과 서버의 디코딩·추론·원본 저장 시간 |
| `request_ms` | 전송 호출부터 응답 처리까지의 시간. 서버 처리도 포함 |
| `capture_to_overlay_ms` | 캡처 시작부터 해당 결과를 화면에 그리기까지의 전체 지연 |
| `overlay_interval_ms`, `previous_overlay_age_ms` | 화면 갱신 간격과 직전 결과의 나이 |

`request_ms - server_ms`는 순수 네트워크 시간이 아닙니다. **테스트 종료 후 저장 완료를 확인하고 페이지를 닫으세요.** 강제 종료하면 아직 전송하지 않은 지연 로그가 유실될 수 있습니다.

### 프레임별 결과 시각화

프레임별 결과 사진도 필요하면 아래 명령을 실행하세요. 원본 `frames/`는 그대로 두고 `annotated/`에 프레임별 사진과 `results.mp4`를 만듭니다. Contact sheet는 저장하지 않습니다. 탐지하지 못한 프레임에는 `NO DETECTION`이 표시됩니다.

```bash
.venv/bin/python scripts/visualize_session.py backend/data/sessions/<세션 ID>
```

여러 세션을 하나의 영상으로 이어 붙일 때는 `--combine` 옵션을 사용합니다. 출력 박스는 `results.jsonl`의 검출 목록을 따릅니다. 신호등의 횡단보도 박스는 내보낸 영상에만 표시되며, 실시간 화면에서는 숨깁니다.

앱 안에 기록 조회 화면은 없습니다. 필요하면 API 로 볼 수 있습니다: `GET /api/sessions`, `GET /api/sessions/{id}`, `GET /api/sessions/{id}/results`, `GET /api/sessions/{id}/frames/{frame_id}.jpg`, `GET /api/sessions/{id}/video`(영상 다운로드). 전체 API 문서는 서버를 켠 상태에서 http://127.0.0.1:8000/docs 입니다.

## 7. Git 사용 규칙

`.gitignore` 로 아래는 **커밋되지 않습니다.** 각자 PC 에만 남습니다.

| 경로 | 내용 |
| --- | --- |
| `backend/models/**` | 가중치 파일 |
| `backend/data/` | 테스트 기록 |
| `.env` | 포트 등 개인 설정 |
| `.venv/`, `check_output/`, `screenshots/` | 가상환경, 점검 결과, 캡처 |

- 가중치는 Git 으로 주고받지 마세요. 용량 때문에 push 가 막힙니다. 공유가 필요하면 드라이브를 쓰세요.
- 추론 코드를 공유할 때는 **본인 기능 변경에 필요한 파일을 함께** 커밋하세요 (`inference/traffic/`, `inference/walking/`, `inference/bus.py`). 관련 테스트·문서 변경도 포함하고, 공통 코드 변경은 팀과 조율하세요.
- 공통 모델 패키지의 버전은 `requirements.txt`에서 관리합니다. GPU용 PyTorch 빌드는 각자의 CUDA 환경에 맞게 설치하세요. 기능별로 추가 패키지가 필요하면 팀과 버전을 조율하세요.
- 공통 코드(API, 저장, 화면)를 고쳐야 할 것 같으면 먼저 팀에 이야기해 주세요.
- 뼈대가 업데이트되면 `git pull` 후 서버를 재시작합니다. 패키지가 추가됐으면 `.venv/bin/pip install -r requirements.txt` 를 다시 실행합니다.

## 8. 문제 해결

| 증상 | 원인과 해결 |
| --- | --- |
| `포트 8000 를 이미 다른 프로그램이 쓰고 있습니다` | 4-1 참고. 이전 서버를 끄거나 `.env` 의 `APP_PORT` 변경 |
| 휴대폰에서 502 Bad Gateway | 서버가 꺼졌거나 터널 포트와 서버 포트가 다릅니다. 서버를 켜고 포트를 맞추세요. |
| 휴대폰에서 주소가 안 열림 | 터널을 다시 실행해 주소가 바뀌었습니다. 터미널의 최신 주소를 쓰세요. |
| "이 브라우저는 카메라를 지원하지 않습니다" | `http://` 주소로 접속했습니다. `https://...trycloudflare.com` 주소로 여세요. |
| "카메라 권한이 거부되었습니다" | iPhone: 설정 → Safari → 카메라 → 허용. Android: 주소창 자물쇠 → 권한 → 카메라 허용. 그 뒤 새로고침 |
| 카카오톡 등 앱 안 브라우저에서 카메라가 안 켜짐 | 주소를 복사해 Safari 나 Chrome 에서 직접 여세요. |
| "이미 실행 중인 세션이 있습니다" | 이전 테스트가 종료되지 않았습니다. 그 휴대폰에서 "테스트 종료"를 누르거나 서버를 재시작하세요. |
| 모델 목록에 본인 가중치가 "(가중치 없음)" 으로 나옴 | 폴더 위치나 확장자를 확인하세요. `check_model.py --list` 로 인식 여부를 볼 수 있습니다. |
| 테스트 시작 시 "로딩 실패: ... load() 에 모델 로딩 코드를 넣어주세요" | 가중치는 인식됐지만 5-3 의 코드를 아직 채우지 않았습니다. |
| 테스트 시작 시 `No module named 'ultralytics'` 등 | 패키지를 시스템 Python 에 설치했습니다. `.venv/bin/pip install ...` 로 다시 설치하세요. |
| 박스 위치가 어긋남 | 좌표가 정규화되지 않았거나 xywh 형식입니다. `check_model.py` 의 출력 이미지로 확인하세요. |
| 추론 시간이 수백 ms 이상 | CPU 로 돌고 있을 가능성이 큽니다. 5-1 의 GPU 확인 명령을 실행하세요. |
| 수신 FPS 가 10 에 못 미침 | 신호등은 5FPS, 나머지는 10FPS가 상한입니다. 지연 로그에서 캡처·요청·그리기 시간을 비교하세요. "왕복" 에는 서버 처리도 포함됩니다. 요청 대기열은 없지만 처리 중에는 이전 마스크가 남습니다. |
| 오른쪽 위 "저장" 이 빨간색 | 디스크가 가득 찼거나 `backend/data` 에 쓰기 권한이 없습니다. |
| 화면을 껐다 켜니 영상이 멈춤 | "카메라 재시작"을 누르세요. 테스트는 유지되고 전송이 이어집니다. |

## 9. 참고

### 설정 (.env)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `APP_HOST` | 127.0.0.1 | 그대로 두세요 (4-4) |
| `APP_PORT` | 8000 | 겹칠 때만 변경 (4-1, 4-2) |
| `DATA_DIR` | backend/data | 기록 저장 위치. 용량이 큰 드라이브로 바꿀 수 있습니다. |
| `MODEL_DIR` | backend/models | 가중치 폴더 |
| `SAVE_FRAMES` | true | false 면 이미지는 저장하지 않고 결과만 저장 |
| `MAX_UPLOAD_BYTES` | 2097152 | 프레임 한 장 업로드 한도 |
| `MIN_FREE_DISK_GB` | 2 | 여유 공간이 이보다 적으면 새 테스트를 막음 |

전송 기본값은 신호등 **960px·5FPS·검출 신뢰도 0.25**, 도보 장애물·버스 **640px·10FPS**입니다. 실제 보행 모델의 검출 신뢰도 기본값은 `config/walking_risk.yaml`의 `yolo.conf`이며, 보행 Mock과 버스의 기본값은 0.4입니다. 촬영·전송 설정은 `backend/static/js/app.js`의 `SETTINGS`와 `TRAFFIC_SETTINGS`에서 관리합니다. 보행 모드의 휴대폰 요청은 신뢰도를 생략하고 서버가 기본값을 정합니다. JPEG 품질은 0.8, 요청 제한 시간은 5초입니다. JPEG 0.8은 인코더 품질 값이며 파일 크기를 80% 또는 0.8%로 고정하는 압축률이 아닙니다.

### 코드 구조

```text
config/                             보행 위험 판단 설정
backend/app/
  ├─ api/                           서버 요청 처리
  ├─ services/                      세션·저장·영상 처리
  │   └─ walking/                   보행 전용 저장·영상 처리
  └─ inference/
      ├─ base.py / registry.py / mock.py  공통 규격·모델 목록·Mock
      ├─ traffic/                  신호등 추론·추적
      ├─ walking/                  장애물 추론·보도 분할
      │   ├─ risk/                 위험 판단
      │   └─ visualization/        화면·결과 영상 표시
      └─ bus.py                    버스 추론
backend/static/                      휴대폰 화면·JS·음성
backend/models/<기능>/               가중치 (Git 제외)
backend/data/sessions/               테스트 기록 (Git 제외)
scripts/                             실행·모델 점검·결과 시각화
tests/backend/ · tests/frontend/     서버·브라우저 테스트
docs/                                기능별 상세 설명
```

### 테스트

```bash
.venv/bin/python -m pytest tests -q
```

브라우저 코드 테스트에는 Node.js 가 필요합니다. 아래 명령은 저장소 루트에서 실행하며, 실제 카메라 없이 검사합니다.

```bash
node tests/frontend/test_capture.cjs
node tests/frontend/test_client_timings.cjs
node tests/frontend/test_recorder.cjs
node tests/frontend/test_guidance.cjs
node tests/frontend/walking_overlay.test.cjs
```

신호등 선택·추적·대상 변경, API·저장 오류, 마스크 픽셀, 캡처 대체 경로, 로그 재시도·안전 종료, 녹화 합성을 검사합니다. 공통 코드를 고쳤다면 push 전에 돌려 주세요.
