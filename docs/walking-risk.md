# 도보 위험 판단 실행·저장 안내

기존 32클래스 YOLO26n에 integration의 보도 ROI·추적·위험 판단을 연결했다. 모델 ID와 YOLO 가중치는 유지한다. 실제 도보 모델을 선택하면 YOLO와 Mask2Former를 한 번 로딩하고, 앱에서 받은 각 프레임의 마스크로 같은 프레임의 ROI와 표면 경고를 판단한다. 화면의 ‘보도 마스크 표시’를 꺼도 보도 판단은 계속된다.

## 환경과 Python 3.12 전환

보행 위험 기능은 Python 3.12 이상을 기준으로 작성했다. 현재 앱의 Python 3.14 환경을 유지했고, 기존 integration Python 3.12 환경에서 실제 모델 두 개, 위험 엔진, 원본/마스크 저장, H.264 영상 출력을 검증했다. 3.12에서 FastAPI 서버 전체를 새 환경에 설치해 실행하는 검증은 하지 않았다.

- 모델 공통 버전: torch 2.14.0+cu130, torchvision 0.29.0+cu130, ultralytics 8.4.152.
- 보도/영상 추가 패키지: `backend/requirements-walking.txt`.
- lap: Python 3.12는 integration과 같은 0.5.12, Python 3.14는 해당 wheel이 있는 0.5.13. 설치 파일의 환경 조건이 자동 선택한다.
- transformers 5.17.0, scipy 1.18.1, PyYAML 6.0.3, Pillow 12.3.0, imageio-ffmpeg 0.6.0.
- 3.14 전용 문법/API를 사용하지 않는다. 3.12 문법 파싱과 위험 회귀 테스트를 수행했다.

추후 Python을 바꿀 때 현재 `.venv`의 Python 실행 파일만 바꾸거나 site-packages를 복사하지 않는다. 원하는 시점에 별도의 Python 3.12 가상환경을 만든 뒤, 해당 환경에 GPU용 torch/torchvision 및 Ultralytics를 설치하고 아래 설치 파일을 적용한다. 이 작업에서 현재 앱 환경을 3.12로 전환하지 않았다.

```bash
# 앞으로 만들어 둔 Python 3.12 환경을 활성화한 상태에서 실행
python -m pip install -r backend/requirements.txt
python -m pip install -r backend/requirements-walking.txt
python -m pytest tests -q
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

가중치 위치:

```text
backend/models/walking/finetune_v2_exp02_stage2_best.pt
backend/models/walking_aux/mask2former/config.json
backend/models/walking_aux/mask2former/preprocessor_config.json
backend/models/walking_aux/mask2former/model.safetensors
```

보조 모델은 모델 선택 목록에 별도로 노출되지 않는다. 다른 GPU/운영체제에서의 torch 설치는 그 환경에 맞춰 준비한다. 공용 requirements 파일은 GPU 패키지를 자동 업그레이드하지 않는다.

## 설정

| 환경변수 | 기본값/의미 |
| --- | --- |
| WALKING_RISK_ENABLED | true. false이면 YOLO의 기존 임시 경고 경로로 복귀 |
| WALKING_RISK_CONFIG | backend/config/walking_risk.yaml |
| WALKING_MASK_WEIGHTS | backend/models/walking_aux/mask2former |
| WALKING_PRECISION | fp32. fp16은 CUDA에서 Mask2Former autocast 사용 |
| WALKING_FONT_PATH | 한글 TTF/TTC 경로. 비어 있으면 Linux Noto CJK 또는 Windows/WSL 맑은 고딕 탐색 |

설정은 서버 재시작 후 적용한다. 기본 YOLO confidence는 도보 0.25, 신호등/버스 0.4다. API 요청에서 confidence를 지정하면 지정한 값을 사용한다. 나머지 YOLO 조건은 imgsz 640, nms=None, iou 0.7, max_det 300, rect=True다. 위험 임계값과 추적 설정은 integration 프로파일을 그대로 이식했다. 모델의 전체 원본 입력을 ROI로 잘라내지 않는다.

FP32가 기본이다. 앱 실측에서 통신·저장 포함 지연이 목표를 넘을 때 FP16 또는 마스크 갱신 주기를 별도로 검토한다. 현재 구현은 오래된 마스크를 새 관측처럼 재사용하지 않는다.

## 표시

보행가능 영역은 초록, 횡단보도는 핑크, 진행 ROI는 청록, 근접 ROI는 자홍이다. 경고는 유지/해제를 반영한 alert_level로 표시한다. 순간 risk_level도 기록한다. 시야에서 사라진 객체는 가짜 박스를 그리지 않고 안내 문구로 표시한다.

한국어 대표 경고는 실시간 캔버스 좌측 상단과 결과 영상 안에 작게 들어간다. 원본 JPEG는 변경하지 않는다. 실시간 WebM은 최신 카메라 영상에 최근 오버레이를 합성한 보조 자료이며, 정확한 프레임 대응은 서버 결과 MP4로 확인한다.

## 필수 저장물

보행 위험 세션은 SAVE_FRAMES=false여도 유효한 입력 JPEG를 추론 전에 바이트 그대로 보존한다. 분석 실패 프레임도 결과 영상에 포함한다. 미전송 카메라 센서 프레임까지 저장하는 기능은 아니다.

```text
<session>/
  manifest.json                  세션 요약, 원본/분석 건수, 모델·설정 출처
  frames/00000001.jpg             수신 원본 JPEG
  inputs/00000001.json            촬영/수신 시각, 원본 크기·해시
  frames.jsonl                   입력 목록
  masks/00000001.png              손실 없는 클래스 인덱스 지도
  risk/00000001.json              전체 위험 판단 또는 실패 정보
  risk.jsonl                     전체 위험 판단 로그
  outcomes/00000001.json          동일 요청 재전송용 결과
  results.jsonl                  기존 API 결과와 오류
  export.json                    pending/running/ready/failed 상태
  result_visualized.mp4           같은 프레임의 마스크·ROI·경고 합성
  result_visualized.frames.json   원본 ID ↔ 출력 PTS·회전 여백 대응
  realtime_overlay.webm           브라우저 녹화가 지원될 때의 보조 자료
```

manifest의 export_status_file과 frame_mapping_file이 출력 상태/대응 파일을 가리킨다. 결과 영상은 촬영 시각 간격으로 재생하며 유효하지 않은 역행·중복 시각은 수신 간격을 사용하고 그 근거를 매핑에 남긴다. 마지막 프레임의 표시 시간을 보존하기 위해 끝점 프레임 하나를 복제하며 매핑에 명시한다. 가로/세로가 바뀌면 첫 프레임 크기의 짝수 출력 캔버스에 비율을 유지해 넣는다.

세션 종료 API는 인코딩을 기다리지 않는다. 화면에서 영상 생성 상태와 다운로드 링크를 확인한다. 인코딩 오류를 세션 원본 저장 완료와 구분한다. 서버가 재시작되면 미완료 세션을 aborted로 정리하고 저장 자료에서 출력을 재개한다. 원본 해시·전체 영상 디코딩·프레임 수·PTS 오차(1.5ms 이하)를 검증한 후 최종 파일을 공개한다. 기존 정상 결과는 덮어쓰지 않는다.

- 조회: GET /api/sessions/{id}의 raw_frame_count, export.
- 다운로드: GET /api/sessions/{id}/result-video.
- 실패 후 재시도: POST /api/sessions/{id}/export.
- 오프라인 재생성: `python scripts/export_walking_session.py <종료된 세션 폴더>`.
- JPEG 검토: `python scripts/visualize_session.py <세션 폴더>`. 보행 위험 기록이 있으면 같은 렌더러를 사용한다.

## 검증 범위

Python 3.14 API/위험/저장/영상 회귀, 3.12 위험 회귀와 실제 모델 정합·영상 출력, JS 캔버스 표시·오래된 콜백·회전·마스크 토글을 검증했다. 신호등 소스와 신호 표시 분기를 바꾸지 않았다.

RTX 5080 Laptop GPU, 실제 저장 프레임 12장, FP32에서 초기화 이후 로컬 HTTP 처리 평균 약 103ms, p95 약 113ms였다. 첫 프레임 추론에는 약 1.15초의 초기화 비용이 있었다. 이 수치는 휴대폰 JPEG 생성·무선 네트워크·터널 지연을 포함하지 않는다. 휴대폰 실시간 5fps와 현장 경고 품질은 실제 연결에서 별도로 확인해야 한다.
