# SafeRoute Raspberry Pi congestion observer

Raspberry Pi가 영상 또는 RTSP를 읽고 로컬 ROI 안의 사람 수를 5초 단위로 집계해 Spring BE에 전달하는 프로젝트다. Python은 밀도, 혼잡 단계, 면적, GridCell 또는 MapEdge를 계산하지 않는다.

## 책임 경계

Python은 person 검출, bottom-center ROI 판정, `avgHeadcount`/`peakHeadcount` 집계와 전송만 담당한다. Spring BE는 `cctvCode`로 CCTV와 FloorGridCell을 조회하고 실제 면적 합계로 밀도를 계산한 뒤 서울시 기준 단계 판정, MapEdge 조회, 경로 재계산, DynamoDB 저장, WebSocket 발행과 유도등 명령 생성을 담당한다. `floor_analysis`와 이 런타임 사이에는 호출이나 패키지 의존성이 없다.

기존 POC는 한 파일에서 YOLO, 첫 프레임 ROI 선택, bounding-box **center**, 임시 `cameraId/zoneId/personCount/congested` 전송을 수행했다. 새 구조는 입력·검출·ROI 저장·집계·reporter·오프라인 큐를 분리하며, 발 위치에 가까워 구역 진입 판단에 더 적합한 **bottom-center**를 기본으로 쓴다. ROI 선 위의 점도 내부다.

## 구조

```text
src/raspberry_pi_congestion/
  detectors/              PersonDetector, fake/ultralytics/onnx, 미완성 Hailo adapter
  video_source.py         FileVideoSource, RtspVideoSource
  roi_provider.py         대화형 4점 선택, 정규화 JSON 저장/로드
  roi_counter.py          bottom-center polygon 판정
  window_aggregator.py    5초 인원 집계와 half-up 정수 평균
  api_client.py           logging/Spring reporter와 설정 가능한 인증 헤더
  offline_queue.py        단일 observation SQLite 큐
  app.py                  파이프라인과 자원 정리
  main.py                 실행 모드 진입점
config/roi/               CCTV별 정규화 ROI JSON
models/                   로컬 추론 모델(파일은 Git 제외)
sample_videos/            로컬 테스트 영상(파일은 Git 제외)
```

Hailo adapter는 설치된 HailoRT 버전, 대상 Hailo 칩, HEF tensor와 후처리가 실기기에서 검증되기 전까지 의도적으로 `HailoAdapterNotImplemented`를 발생시킨다. 동작 완료 상태가 아니다. 일반 프레임과 모델을 EC2에 저장하지 않으며 S3 업로드도 아직 구현하지 않는다. 향후 이벤트 스냅샷이 필요하면 observation의 `s3ImageKey`를 채우는 업로더를 파이프라인 경계에 추가한다.

## 설치와 PyCharm

PyCharm에서 이 디렉터리를 프로젝트로 열고 Python 3.9 이상 venv를 만든다.

```powershell
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

Run Configuration은 module `raspberry_pi_congestion.main`, parameters는 아래 모드 중 하나, working directory는 이 디렉터리로 둔다. `.env.example`을 `.env`로 복사한 뒤 자리표시자만 로컬 값으로 바꾼다. `.env`는 커밋하지 않는다.

## ROI 설정과 실행

GUI가 있는 개발 PC에서 영상 첫 프레임의 네 점을 순서대로 클릭한다. 좌표는 0~1 JSON으로 저장되며 운영 Raspberry Pi는 이 파일만 읽으므로 GUI가 필요 없다. Esc는 취소다.

```powershell
$env:VIDEO_SOURCE='{VIDEO_FILE_PATH}'
$env:CCTV_CODE='{CCTV_CODE}'
python -m raspberry_pi_congestion.main setup-roi
```

`ROI_CONFIG_PATH`를 생략하면 `CCTV_CODE=CCTV_TEST_01` 기준으로
`.\config\roi\CCTV_TEST_01.json`에 저장한다. 여러 CCTV는 파일을 각각 분리한다.

로컬 영상 dry-run은 서버로 보내지 않고 안전한 payload 로그만 남긴다.

```powershell
$env:VIDEO_SOURCE='{VIDEO_FILE_PATH}'
$env:CCTV_CODE='{CCTV_CODE}'
$env:DETECTOR_BACKEND='ultralytics'
$env:MODEL_PATH='.\models\yolov8n.pt'
python -m raspberry_pi_congestion.main dry-run
```

`MODEL_PATH`는 실행 시점의 working directory 기준이다. 위 명령처럼 프로젝트 루트에서
실행하고 모델이 `models` 폴더에 있다면 `.\models\yolov8n.pt`가 정확하다. `MODEL_PATH`를
생략한 Ultralytics 모드도 기본적으로 `.\models\yolov8n.pt`를 찾으며, 파일이 없으면
Ultralytics가 다운로드를 시도할 수 있다. 모델 파일과 `models/`는 Git에서 제외된다.

파일을 서버로 전송하려면 마지막 인자를 `file`로 바꾸고 서버 변수를 설정한다. RTSP는 `VIDEO_SOURCE='{RTSP_URL}'`로 설정하고 `rtsp` 모드를 사용한다. URL에 포함된 비밀번호와 인증 토큰은 로그에 출력하지 않는다.

```powershell
$env:SAFEROUTE_SERVER_BASE_URL='{SERVER_BASE_URL}'
$env:CONGESTION_OBSERVATION_PATH='{API_PATH}'
$env:DEVICE_AUTH_TOKEN='{DEVICE_AUTH_TOKEN}'
python -m raspberry_pi_congestion.main file
python -m raspberry_pi_congestion.main rtsp
```

`test` 모드는 `DETECTOR_BACKEND=fake`와 함께 모델 없이 파이프라인을 확인할 때 사용한다.

## 임시 수신 서버

최종 Spring API가 구현되기 전에는 `LoggingCongestionReporter` 또는 개발용 Flask 수신기로 확인한다. 이 mock은 Spring API 구현 또는 실제 연동 성공을 의미하지 않는다.

```powershell
pip install flask
python scripts/mock_server.py
$env:SAFEROUTE_SERVER_BASE_URL='http://127.0.0.1:8080'
$env:CONGESTION_OBSERVATION_PATH='/api/v1/device/congestion-observations'
python -m raspberry_pi_congestion.main file
```

인증 헤더 명칭은 미확정이다. `AUTH_HEADER_NAME`과 `AUTH_HEADER_PREFIX`로 조정하며 기본값은 `Authorization: Bearer ...`다.

## 최종 요청

```json
{
  "eventId": "{UUID}",
  "cctvCode": "{CCTV_CODE}",
  "avgHeadcount": 5,
  "peakHeadcount": 8,
  "sampleCount": 25,
  "windowStart": 1786500000000,
  "windowEnd": 1786500005000,
  "capturedAt": 1786500005000,
  "s3ImageKey": null
}
```

평균은 `floor(value + 0.5)` half-up 정수이며 빈 window는 전송하지 않는다. HTTP는 2xx를 성공으로 보고 400/401/403/404는 재시도하지 않는다. 429, 5xx, 네트워크 오류만 지수 backoff로 제한 재시도한다. 실패 observation은 최대 1,000건/24시간의 SQLite 큐에 보관하고 기본 30초 간격으로만 flush한다.

## Spring BE에 아직 필요한 것

- `POST /api/v1/device/congestion-observations` (경로 최종 합의 필요)
- 위 JSON과 정확히 일치하는 Integer `avgHeadcount`/`peakHeadcount` 요청 DTO
- `eventId` 기반 멱등 처리
- 디바이스 인증 헤더 방식의 최종 합의와 필터
- `cctvCode` 조회 실패 및 DTO 검증 오류 응답 정책
- GridCell 면적 기반 density/서울시 단계/MapEdge 이후 처리 구현
