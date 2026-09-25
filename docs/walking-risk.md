# 도보 위험 판단 실행·저장 안내

기존 32클래스 YOLO26n에 integration의 보도 ROI·추적·위험 판단을 연결했다. 모델 ID와 YOLO 가중치는 유지한다. 실제 도보 모델을 선택하면 YOLO와 Mask2Former를 한 번 로딩하고, 앱에서 받은 각 프레임의 마스크로 같은 프레임의 ROI와 표면 경고를 판단한다. 화면의 ‘보도 마스크 표시’를 꺼도 보도 판단은 계속된다. 아래의 테스트 수치와 지연 측정은 당시 기록이며 현재 코드의 재검증 결과를 뜻하지 않는다.

## Python 3.12 환경

보행 위험 기능의 기준 버전은 Python 3.12다. 기존 integration Python 3.12 환경에서 실제 모델 두 개, 위험 엔진, 원본/마스크 저장, H.264 영상 출력을 검증했다.

- 모델 공통 버전: torch 2.14.0+cu130, torchvision 0.29.0+cu130, ultralytics 8.4.152.
- 서버·테스트·보도/영상 패키지: `requirements.txt`.
- lap: Python 3.12는 integration과 같은 0.5.12, Python 3.14는 해당 wheel이 있는 0.5.13. 설치 파일의 환경 조건이 자동 선택한다.
- transformers 5.17.0, scipy 1.18.1, PyYAML 6.0.3, Pillow 12.3.0, imageio-ffmpeg 0.6.0.

다른 Python 버전의 가상환경을 3.12로 전환할 때는 실행 파일이나 site-packages를 교체하지 말고 Python 3.12로 새 가상환경을 만든다. 그 환경에 GPU용 torch/torchvision 및 Ultralytics를 설치하고 아래 설치 파일을 적용한다.

```bash
# Python 3.12 환경을 활성화한 상태에서 실행
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

가중치 위치:

```text
backend/models/walking/finetune_v2_exp02_stage2_best.pt
backend/models/walking/mask2former_w/config.json
backend/models/walking/mask2former_w/preprocessor_config.json
backend/models/walking/mask2former_w/model.safetensors
```

보조 모델은 모델 선택 목록에 별도로 노출되지 않는다. 다른 GPU/운영체제에서의 torch 설치는 그 환경에 맞춰 준비한다. 공용 requirements 파일은 GPU 패키지를 자동 업그레이드하지 않는다.

## 설정

| 환경변수 | 기본값/의미 |
| --- | --- |
| WALKING_RISK_ENABLED | true. 위험 판단의 유일한 켜기/끄기 설정. false이면 YOLO의 기존 임시 경고 경로로 복귀 |
| WALKING_RISK_CONFIG | config/walking_risk.yaml |
| WALKING_MASK_WEIGHTS | backend/models/walking/mask2former_w |
| WALKING_PRECISION | fp32. fp16은 CUDA에서 Mask2Former autocast 사용 |
| WALKING_FONT_PATH | 한글 TTF/TTC 경로. 비어 있으면 Linux Noto CJK 또는 Windows/WSL 맑은 고딕 탐색 |

설정은 서버 재시작 후 적용한다. `config/walking_risk.yaml`의 `risk`에는 위험 판단 규칙만 두고, 켜기/끄기는 `.env`의 `WALKING_RISK_ENABLED`로 관리한다. 위험 판단이 켜진 실제 보행 모델의 기본 YOLO confidence는 `config/walking_risk.yaml`의 `yolo.conf`에서 읽는다. 휴대폰의 보행 모드는 세션 요청에 confidence를 넣지 않으며, API 요청에서 confidence를 명시하면 요청값을 우선한다. 신호등 기본값은 0.25, 버스는 0.4다. 보행 Mock은 YOLO를 실행하지 않고 공통 세션 기본값 0.4를 사용한다.
도보 전송 기본값은 긴 변 640px·최대 10FPS이며 `backend/static/js/app.js`의 `SETTINGS`에서 관리한다. 나머지 YOLO 조건은 imgsz 640, nms=None, iou 0.7, max_det 300, rect=True다. 위험 임계값과 추적 설정은 integration 프로파일을 그대로 이식했다. 모델의 전체 원본 입력을 ROI로 잘라내지 않는다.

FP32가 기본이다. 앱 실측에서 통신·저장 포함 지연이 목표를 넘을 때 FP16 또는 마스크 갱신 주기를 별도로 검토한다. 현재 구현은 오래된 마스크를 새 관측처럼 재사용하지 않는다.

2026-09-23 위험 정책 이식: 하단 x=0.02–0.98, y=0.75–1.00은 고정하고 상단 경로는 연결된 보행 마스크에 따라 y=0.30–0.60으로 조정한다. 고정 장애물의 주의·위험 기준은 각각 화면 y=0.55/0.78이다. 넓은 하단 ROI의 측면 물체는 겹침만으로 위험이 되지 않으며 중앙 접근·극근접·짧은 TTC를 별도로 평가한다. 0.5초 초과 간격에서는 운동 이력을 무효화하고 경고 상태를 유지하며, 2.0초 초과 간격에서는 전체 상태를 초기화한다. 대표 경고는 위험도와 접근 근거를 우선하며, 클래스가 안정적으로 확인되기 전에는 일반 장애물로 표시한다. 원본 YOLO 클래스는 기록에 보존한다.

## 표시

보행가능 영역은 초록, 횡단보도는 핑크, 진행 ROI는 청록, 근접 ROI는 자홍이다. 경고는 유지/해제를 반영한 alert_level로 표시한다. 마스크는 기존 보행 영역 기능과 같은 `event.mask_rle`로 보내고 구간이 많은 마스크만 `mask_png`로 대체한다. 실시간 화면은 그 마스크 위에 ROI와 경고를 겹쳐 그린다. 순간 risk_level도 기록한다. 시야에서 사라진 객체는 가짜 박스를 그리지 않고 안내 문구로 표시한다.

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

## 이전 검증 기록

Python 3.14 API/위험/저장/영상 회귀, 3.12 위험 회귀와 실제 모델 정합·영상 출력, JS 캔버스 표시·오래된 콜백·회전·마스크 토글을 검증했다. 신호등 소스와 신호 표시 분기를 바꾸지 않았다.

RTX 5080 Laptop GPU, 실제 저장 프레임 12장, FP32에서 초기화 이후 로컬 HTTP 처리 평균 약 103ms, p95 약 113ms였다. 첫 프레임 추론에는 약 1.15초의 초기화 비용이 있었다. 이 수치는 휴대폰 JPEG 생성·무선 네트워크·터널 지연을 포함하지 않으며, 이번 위험 정책 이식 전의 측정값이다. 새 정책의 현장 경고 선행 시간과 휴대폰 실시간 5fps 달성 여부는 실제 연결에서 별도로 확인해야 한다.

## 현재 앱 설정을 정한 기록 (2026-09-23)

`main`의 실시간 지연 개선·탐지 영상 자동 저장·신호등 음성 안내와 합치면서 아래를 정리했다.
프로젝트 전체 실행 방법은 README에, 보행 위험 판단의 세부 설정은 이 문서에 기록한다.

| 항목 | 정한 값 | 이유 |
| --- | --- | --- |
| 전송 해상도 | 긴 변 640px (현재 앱 기본값) | YOLO 입력이 640, 보도 마스크 입력이 384로 고정이라 960 업로드는 판단에 기여하지 않는다 |
| 전송 상한 | 10FPS (현재 앱 기본값) | 0.6초 운동 이력의 표본이 늘고 프레임 간 카메라 이동량이 줄어 TTC·측방 진입 판단에 유리하다 |
| 검출 신뢰도 | `config/walking_risk.yaml`의 `yolo.conf` (현재 0.25) | 값을 올리면 검출이 줄어 경고 미발생이 늘어난다 |
| 마스크 전송 | `mask_rle` 우선, `mask_png` 대체 | main의 표시 지연 개선을 유지한다 |
| 결과 영상 | 보행 위험 세션은 `result_visualized.mp4`만 생성 | 고정 속도 영상(`annotated/results.mp4`)은 촬영 간격을 재현하지 못한다. 보행 세션의 `video_status`는 `walking_export`로 남는다 |

이 문서에는 640px에서 볼라드·연석처럼 작은 장애물의 검출 변화를 실촬영으로 확인한 결과가 없다.
위의 지연·정합 측정은 당시 960px 프레임에서 얻은 값이다.


## 촬영 불가 상태 (2026-09-23)

`risk.camera_view_guard_enabled=true`로 integration의 영상 기반 촬영 상태 판정을 사용한다. 바닥 위주의 화면과 화면 대부분을 차지하는 비정상 검출이 함께 나타나거나 렌즈 가림·심한 화질 저하가 지속되면 `camera_view.status=unavailable`이 된다. 이때 원본 검출은 기록하되 새 객체별 위험 경고와 ROI·검출 박스는 라이브 화면과 결과 영상에서 숨기고 “주의 · 촬영 불가 · 카메라를 전방으로 들어 주세요”를 표시한다. 이미 확인된 위험은 최대 0.8초 동안 출처를 명시해 유지한다. 시야가 읽히고 카메라 움직임이 0.35초 안정되면 자동으로 복구하며 추적·ROI 상태를 다시 시작한다. 버튼 조작은 필요하지 않다.

API `event.camera_view`에는 상태·원인·증거가, `event.risk_events`에는 상태 전환이 기록된다. 촬영 불가 중 검출의 기하 위험 원본은 `detections[].extra.untrusted_risk_level`로 남긴다. 현재 도보 음성 안내는 브라우저의 `guidance.js`가 `walking_warning` 중 `level=danger`와 `voice_category`가 있는 이벤트만 읽어 사람·차량·장애물 음원을 재생한다. 같은 이벤트·분류의 안내는 5초 안에 반복하지 않으며, 촬영 불가 상태 자체를 별도 음성으로 안내하지 않는다. 영상만으로 절대 휴대폰 각도를 확인할 수 없으므로 이 상태를 전방 주시의 증명으로 사용하지 않는다.
