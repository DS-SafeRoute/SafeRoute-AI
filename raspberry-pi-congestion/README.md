# SafeRoute Raspberry Pi congestion observer

Raspberry Pi가 CCTV 영상의 ROI 안 사람 수를 5 FPS로 추론하고, Spring BE 설정에 따라 관측값·혼잡 이벤트·JPEG 이미지를 전송한다. Pi는 `GridCell`, `MapEdge`, 경로 계산, 관리자 승인 처리를 하지 않는다.

## 현재 연동 계약

- 장치 코드는 데모 기준 `CCTV_001` 또는 `CCTV_002`이고, 각 장치의 전용 Bearer Token은 `DEVICE_AUTH_TOKEN`으로만 주입한다.
- `GET /api/v1/device/congestion-config?cctvCode=...`를 훈련 중 5초, 비활성 중 15초 간격으로 조회한다.
- `trainingSessionId`는 BE가 준 UUID를 그대로 사용한다. Pi가 세션 ID를 생성하지 않는다.
- `trainingActive=false`이면 추론, 관측값, 이벤트, 이미지 인코딩·업로드, Presigned URL 요청을 중단한다.
- `configVersion` 또는 세션/활성 상태가 바뀌면 집계 창, 추론 FPS, 임계값과 이벤트 설정을 즉시 적용한다.
- 밀도는 `headcount / monitoredAreaM2`로 계산하고 단계 임계값은 BE 응답만 사용한다.
- 혼잡 진입/상승은 기본 3프레임, 정상 복귀는 5프레임 연속 조건이며 단계 상승은 cooldown과 무관하게 즉시 보낸다.
- 모든 시간 필드는 Unix timestamp 밀리초다.

5초 관측값의 `avgHeadcount`는 정확도를 위해 실수로 보낸다. `peakHeadcount`와 `sampleCount`는 정수다. 이미지 업로드가 실패해도 `monitoringImageKey: null`로 관측값은 전송한다.

```json
{
  "eventId": "observation-uuid",
  "trainingSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "cctvCode": "CCTV_001",
  "avgHeadcount": 4.75,
  "peakHeadcount": 8,
  "sampleCount": 25,
  "windowStart": 1786500000000,
  "windowEnd": 1786500005000,
  "capturedAt": 1786500005000,
  "monitoringImageKey": null,
  "configVersion": 1
}
```

혼잡 이벤트에는 `edgeId`와 `eventImageKey`를 넣지 않는다. 이벤트 POST와 이미지 업로드를 병렬 처리한 뒤, 둘 다 성공하면 `PATCH /api/v1/device/congestion-events/{eventId}/image`로 BE가 발급한 `objectKey`를 연결한다.

## 설치 및 실행

Python 3.9 이상 가상환경에서 다음을 실행한다.

```powershell
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

`.env.example`을 `.env`로 복사하고 실제 값으로 바꾼다. 토큰과 RTSP 비밀번호는 로그에 출력하지 않으며 `.env`는 커밋하지 않는다.

```powershell
$env:CCTV_CODE='CCTV_001'
$env:DEVICE_AUTH_TOKEN='{CCTV_001 전용 토큰}'
$env:SAFEROUTE_SERVER_BASE_URL='https://{BE_HOST}'
$env:VIDEO_SOURCE='{VIDEO_FILE_OR_RTSP_URL}'
python -m raspberry_pi_congestion.main file
# 또는
python -m raspberry_pi_congestion.main rtsp
```

## Hailo NPU 실행

현재 Hailo backend는 HailoRT 4.x 기반 Hailo-8/Hailo-8L과 다음 계약의
객체검출 HEF를 지원한다.

- 단일 NHWC RGB 입력(`uint8`)
- `HAILO_NMS_BY_CLASS` 형식의 단일 출력
- COCO class 순서(`person` class id `0`)
- NMS row: `[ymin, xmin, ymax, xmax, score]` 정규화 좌표

일반 `yolov8n.pt`는 Hailo에서 실행할 수 없다. 장치 아키텍처와 설치된
HailoRT 버전에 맞는 `.hef`를 사용해야 한다. Hailo-10H는 현재 지원 범위가
아니며 별도의 HailoRT 5.x 어댑터 검증이 필요하다.

AI Kit 또는 Hailo-8/8L AI HAT+에서 Hailo 패키지가 아직 없다면 Raspberry Pi
OS의 공식 패키지를 설치하고 재부팅한다.

```bash
sudo apt update
sudo apt install -y dkms hailo-all
sudo reboot
```

장치와 모델을 확인한다.

```bash
hailortcli fw-control identify
dmesg | grep -i hailo
find /usr/share/hailo-models -type f -name '*.hef' 2>/dev/null
hailortcli run /실제/모델/경로/model.hef
```

APT로 설치한 `hailo_platform`이 기존 가상환경에서 보이지 않으면 시스템
site-packages를 사용하는 가상환경을 만든 뒤 Pi 의존성을 다시 설치한다.

```bash
python3 -m venv --system-site-packages .venv-hailo
source .venv-hailo/bin/activate
pip install -r requirements-pi.txt
pip install -e . --no-deps
python -c "import hailo_platform; print('hailo_platform import 성공')"
```

서버 전송 없이 CCTV별로 먼저 검증한다.

```dotenv
RUN_MODE=dry-run
CCTV_CODE=CCTV_001
VIDEO_SOURCE=rtsp://{USER}:{PASSWORD}@{CCTV_IP}:554/{STREAM_PATH}
ROI_CONFIG_PATH=./config/roi/CCTV_001.json
DETECTOR_BACKEND=hailo
MODEL_PATH=/실제/모델/경로/model.hef
DETECTOR_CONF_THRESHOLD=0.4
TARGET_INFERENCE_FPS=5
SHOW_PREVIEW=false
```

```bash
python -m raspberry_pi_congestion.main dry-run
```

5초 관측 로그의 `sampleCount`가 약 25이면 유효 처리 속도가 약 5 FPS다.
`CCTV_001`, `CCTV_002`를 각각 검증한 뒤 `RUN_MODE=rtsp`로 백엔드 통합
테스트를 수행한다. RTSP URL, 카메라 비밀번호, 장치 토큰은 커밋하지 않는다.

개발 PC에서 추론 화면을 확인하려면 `SHOW_PREVIEW=true`를 설정한다. 노란색은 ROI,
초록색 박스는 ROI 안에서 집계된 사람, 주황색 박스는 ROI 밖 사람이다. 창에서 `Q` 또는
`Esc`를 누르면 파이프라인과 미리보기 창이 함께 종료된다. 기본값은 `false`이며 전송용
JPEG에는 오버레이가 포함되지 않는다.

ROI는 CCTV별 파일로 저장한다. `ROI_CONFIG_PATH`를 생략하면 `./config/roi/CCTV_001.json`처럼 장치 코드 기반 경로를 사용한다.

```powershell
python -m raspberry_pi_congestion.main setup-roi
```

서버 없이 검출/집계를 확인하려면 `dry-run` 또는 `test` 모드를 사용한다. 이 두 모드만 로컬 기본 설정을 사용하며 운영 모드는 반드시 BE 설정을 조회한다.

## 재시도와 로컬 큐

200/201을 포함한 모든 2xx는 성공이다. 5xx, timeout, 429는 동일한 `eventId`, 세션, 시간, 버전, payload로 제한 재시도한 뒤 SQLite 큐에 보관한다. 400/401/403 같은 영구 오류는 큐에 넣지 않는다. 이벤트 이미지 연결의 404/409는 이벤트 생성 순서를 고려해 재시도한다. Presigned URL은 큐에 저장하거나 재사용하지 않는다.

SQLite 큐는 관측값, 혼잡 이벤트, 이벤트 이미지 연결 작업을 구분해서 저장한다. `trainingActive=false`일 때 종료 세션의 기존 큐를 자동 전송하지 않으며, 최종 처리 정책은 BE와 별도 합의가 필요하다.

## 주요 구조

```text
src/raspberry_pi_congestion/
  api_client.py          설정/관측/이벤트/Presigned URL/이미지 연결 API
  app.py                 polling, 추론, 병렬 이벤트·이미지 처리, 큐 재전송
  event_detector.py      연속 프레임과 cooldown 상태 머신
  models.py              최종 BE DTO와 설정 모델
  offline_queue.py       작업 유형별 SQLite 실패 큐
  window_aggregator.py   실수 평균·최대·성공 프레임 수 집계
```

Hailo 코드는 장치가 없는 CI에서 mock으로 검증한다. 실제 호환성, 처리 속도와
장시간 안정성은 대상 라즈베리파이·HailoRT·HEF 조합에서 별도로 검증해야 한다.
