# System integration — Phase 3 hardware boundaries

## 현재 실제 시스템 전제

현재 conveyor에는 물체 진입 sensor와 검사 위치 sensor가 없다. 따라서 production v0 cycle은 사용자가 물체를 올린 뒤 명시적으로 시작하며 `WAIT_OBJECT`에서 sensor를 기다리지 않는다. 현재 흐름은 다음과 같다.

```text
INITIALIZING → READY → CONVEYOR_TO_INSPECTION
→ STRUCTURED_LIGHT_SCAN → PLAN_POSES → pose inspection loop
→ FINALIZE → CONVEYOR_OUT → COMPLETE → STOPPED
```

`WAIT_OBJECT` enum은 향후 sensor hook 및 호환성을 위해 남아 있지만 현재 `run_once()`에서는 사용하지 않는다.

## Conveyor production v0

Arduino serial은 115200 baud이며 다음 legacy command만 지원한다.

```text
F<steps>\n
B<steps>\n
```

완료는 `=== STOP (Target Reached) ===` 출력 중 `Target Reached` substring으로 판정한다. 검사 위치와 배출 위치는 sensor feedback이나 실제 거리가 아니라 시작점 기준 open-loop step count다. Port, 검사/배출 방향, step 수와 timeout은 모두 config로 주입하며 calibration 전 production default는 없다.

현재 firmware는 `for` loop와 `delayMicroseconds` 기반 blocking step generation을 사용한다. 이동 중 새 command나 상태 request를 즉시 처리할 수 없으며 STOP, ESTOP, STATUS protocol도 없다. Host controller는 이 명령을 만들어내지 않는다. 향후 firmware에서는 `ARRIVED`, `MOVING`, `STOPPED` 같은 machine-readable protocol과 위치 sensor/정지 기능을 추가할 수 있다.

실제 conveyor 이동은 사용자가 다음 diagnostic을 직접 실행할 때만 발생한다.

```bash
python3 src/tools/test_conveyor_serial.py --port /dev/ttyACM1 --direction F --steps 1000
```

## 실제 STM serial 경계

`SerialPlatformController`는 115200 8N1 transport, pose packet write, telemetry read 및 stable 대기를 구현한다. Constructor는 port를 열지 않으며 자동으로 `RST`를 보내지 않는다. 기본 diagnostic은 다음과 같이 telemetry만 읽고 어떠한 명령도 송신하지 않는다.

```bash
python3 src/tools/read_platform_telemetry.py --port /dev/ttyACM0 --count 10
```

실제 motion은 verified limit와 safe Z가 아직 없으므로 system hardware factory에 연결하지 않았다.

## Orbbec camera 경계

`OrbbecCameraController`는 기존 Gemini 336L profile/config/aligned RGB+Depth helper를 lifecycle API(`start/capture/close`)로 감싼다. 결과는 `RGBDepthFrame(color_bgr, depth_mm, timestamp)`이다. 실제 surface 계산은 계속 `src.core`가 담당한다.

```bash
python3 src/tools/test_orbbec_camera.py --output-dir /tmp/orbbec_test
```

이 명령은 사용자가 직접 실행할 때만 camera를 연다. 현재 controller는 검증된 helper를 재사용하므로, 다음 단계에서 helper를 experiment script 밖의 camera module로 완전히 추출해야 한다.

## Structured Light shell 경계

원본 `서영 파트 파일`은 수정하지 않는다. `ShellStructuredLightRunner`는 subsystem root, result root와 timeout을 config로 받고 `물체검사.sh`를 실행할 수 있다. 실행 전 script와 result directory, script 내부 `BASE` absolute path를 검사한다.

```bash
python3 src/tools/check_structured_light.py \
  --subsystem-root '/path/to/서영 파트 파일' \
  --result-root '/path/to/샘플'
```

기본은 preflight only다. 실제 실행에는 `--execute`가 명시적으로 필요하다. 현재 repository 복사본은 `/home/seoyeong/...`를 가리키므로 이 PC에서 preflight가 실패하며, 원본을 자동 수정하거나 우회하지 않는다.

## Phase 2/3 범위

Mock workflow는 실제 카메라, 구조광 장치, STM32 및 컨베이어를 작동시키지 않고 전체 workflow의 계약과 조립을 검증한다. 실행 진입점은 다음과 같다.

```bash
python3 src/run_system.py --mode mock --once
```

출력의 `MOCK_COMPLETE`, `MOCK_NORMAL`, `mock: true`는 실제 품질 판정이 아님을 뜻한다. `--mode hardware`는 mock으로 자동 대체되지 않으며 명시적인 오류를 반환한다.

## Architecture

`SystemController`만 전체 `SystemState`를 변경한다. Factory가 conveyor, structured-light runner/adapter, pose planner, platform, Z quality sampler, surface inspector, anomaly detector를 주입한다.

```text
Conveyor → Structured Light → Pose plan
         → safe pose → Automatic Z → Surface inspection
         → Anomaly inspection → final result → Conveyor out
```

Phase 2의 모든 실제 장치 구현은 mock 또는 interface/skeleton이다.

## STM32 protocol

USART2 설정은 115200 baud, 8 data bits, no parity, 1 stop bit, no hardware flow control이다. 명령은 CR/LF로 끝난다.

```text
Z:<float>
R:<float>
P:<float>
Z:<float> R:<float> P:<float>
RST
MODE:0|1|2
```

firmware에 없는 STOP/ESTOP/ABORT 명령은 제공하지 않는다. `PlatformLimits`는 host-side validation의 구조만 제공하며 hardware 기본값은 모두 `None`이다.

Telemetry 형식:

```text
TLM:Z=...,R=...,P=...,S=...,M1=...,M2=...,M3=...,H=...,G=...,C=...,VR=...,VP=...
```

`S=1`은 목표 위치가 외부 센서로 측정되었다는 뜻이 아니라 firmware의 stable/sleep 조건이 성립했다는 뜻이다. `PlatformTelemetry.stable`로 표현한다.

Telemetry의 Z는 encoder 기반 physical measured Z가 아니다. firmware가 nominal motor speed와 경과 시간을 적분해 계산한 controller-estimated Z이다. Z 값은 후보 명령과 복귀 위치 식별에 쓰며, 실제 검사 가능 여부는 RGB/depth quality gate가 판단해야 한다.

## Structured Light artifact contract

Adapter는 알려진 이름을 명시적으로 분류한다.

- `03_*물체만.ply`, `FINAL_DC_MASK_PHASE*.ply`: `OBJECT_ONLY`
- `*물체+플랫폼.ply`: `OBJECT_AND_PLATFORM`
- `*_WITH_FLOOR.ply`: `OBJECT_PLATFORM_FLOOR`
- 그 외: `UNKNOWN`

UNKNOWN을 OBJECT_ONLY로 승격하지 않는다. 기본 선택은 object-only 우선이며, 사용 가능한 전체 artifact 목록도 result metadata에 남긴다. 현재 parser는 ASCII PLY만 지원하고 binary PLY에는 `UnsupportedPLYFormatError`를 발생시킨다.

## Coordinate convention

확인된 writer 식은 다음과 같다.

```text
X = x_pixel - image_width / 2
Y = image_height / 2 - y_pixel
Z = z_sign * z_scale * phase_surface[y, x]
```

X는 영상 중심 기준 오른쪽이 양수, Y는 위쪽이 양수이며 XY 단위는 pixel이다. Z 단위는 `phase_relative`이고 mm가 아니다. `z_scale`, `z_sign`, image width/height가 artifact에서 확인되지 않으면 `None`으로 유지한다.

## Pose 및 Automatic Z

외부 `0823_test.py`의 pose 계산은 relative-phase Z의 물리 단위, STM 부호, absolute Z 의미가 검증되지 않아 production planner로 복사하지 않았다. Phase 2는 metadata에 `source=mock`을 갖는 두 pose만 사용한다.

Automatic Z는 입력 sample을 모두 확인한 뒤 `gate_passed=True`이며 `quality_score`가 가장 큰 후보를 선택한다. 동점이면 낮은 Z를 선택한다. 유효 후보가 없으면 `NoValidInspectionZ` 실패 결과를 반환한다. Phase 2 score와 mock platform limit/safe Z는 테스트 전용이며 실제 calibration 값이 아니다.

## Error and result contract

성공과 실패 모두 `SystemInspectionResult`로 반환된다. 실패 시 `failed_state`, `error_type`, `error_message`, 완료된 pose 결과와 state history를 보존하고 `ERROR → STOPPED`로 종료한다.

## Hardware 미연결 영역과 다음 단계 조건

다음 항목은 아직 구현되지 않았다.

- serial port open/read/write 및 실제 platform motion
- 실제 구조광 shell 실행
- 실제 PLY pose planner
- Orbbec capture를 service로 연결한 surface inspection
- surface-only AE inference와 offline threshold artifact
- 실제 conveyor protocol

Phase 3 전에 실제 Z/roll/pitch limit, safe Z, pose 부호, 구조광 설치 경로와 artifact manifest, conveyor protocol을 장비 담당자와 확정해야 한다.
