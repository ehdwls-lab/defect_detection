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

Port open 직후에는 Arduino reset으로 출력되는 startup banner의 `DUAL STEP MOTOR TEST MODE`,
`Usage`, 두 `Example` 줄과 마지막 separator를 모두 읽어 readiness를 확인한다. 이 handshake가
startup timeout 안에 끝나지 않으면 command를 전송하지 않는다. Readiness 이후 RX를 reset하고
command를 쓴다.

완료는 현재 command의 정확한 firmware echo(`>>> DUAL FORWARD <steps> steps` 또는
`<<< DUAL BACKWARD <steps> steps`)를 먼저 받은 뒤, 그 이후
`=== STOP (Target Reached) ===`의 `Target Reached` substring을 받아야 인정한다. command 직전
RX double-reset과 command-specific echo 경계를 함께 사용하므로 늦게 도착한 과거 completion만으로
다음 단계로 진행하지 않는다. 검사 위치와 배출 위치는 sensor feedback이나 실제 거리가 아니라
encoder-based closed-loop step motor를 사용하지만 conveyor 검사 위치에는 외부 위치 sensor가 없으므로,
시작점 기준 pulse-count positioning이다. Port, 검사/배출 방향, step 수와 timeout은 모두 config로
주입하며 calibration 전 production default는 없다.

현재 firmware는 `for` loop와 `delayMicroseconds` 기반 blocking step generation을 사용한다. 이동 중 새 command나 상태 request를 즉시 처리할 수 없으며 STOP, ESTOP, STATUS protocol도 없다. Host controller는 이 명령을 만들어내지 않는다. 향후 firmware에서는 `ARRIVED`, `MOVING`, `STOPPED` 같은 machine-readable protocol과 위치 sensor/정지 기능을 추가할 수 있다.

실제 conveyor 이동은 사용자가 다음 diagnostic을 직접 실행할 때만 발생한다.

```bash
python3 src/tools/test_conveyor_serial.py --port /dev/ttyACM1 --direction F --steps 1000
```

## 실제 STM serial 경계

`SerialPlatformController`는 115200 8N1 transport, pose packet write, telemetry read 및 stable 대기를 구현한다. Constructor는 port를 열지 않으며 자동으로 `RST`를 보내지 않는다. 기본 diagnostic은 다음과 같이 telemetry만 읽고 어떠한 명령도 송신하지 않는다.

## STM32 motion diagnostic 순서

`src/tools/test_platform_motion.py`는 `SystemController`와 분리된 좌표계 검증 도구다.
기본 실행은 stale RX를 버린 뒤 telemetry 한 건만 읽는 fresh read-only 동작이며, `--execute` 또는
`--execute-pose`와 터미널의 `EXECUTE` 확인 없이는 명령을 쓰지 않는다.

실물 검증 순서는 다음과 같다.

1. `--port /dev/ttyACM0`만 사용해 telemetry를 확인한다.
2. 사용자가 실물 간섭을 측정해 safe Z를 결정한다. 코드나 구조광 Z를 사용하지 않는다.
3. 사용자가 정한 작은 값으로 `--z <value> --execute`를 시험한다.
4. safe height에서 `--roll <positive-small> --execute --ack-safe-height`를 실행하고 방향을 기록한다.
5. Roll을 0, negative-small, 0 순으로 시험한다.
6. Pitch도 positive-small, 0, negative-small, 0 순으로 시험한다.
7. `positive_roll_observed_direction`과 `positive_pitch_observed_direction`을 작업 기록에 남긴다.
8. 구조광 예측 Roll/Pitch와 실제 축 부호를 비교한다.
9. 검증이 끝난 뒤에만 RealPosePlanner의 motion permission 변경을 논의한다.

각 명령은 absolute target이다. orientation에는 `--ack-safe-height`가 필요하다.
pose JSON 실행을 명시적으로 시험할 때도 사용자가 `--z`를 제공해야 하며, host는
Z 전송 → stable 확인 → Roll/Pitch 전송 → stable 확인 순서를 유지한다. JSON의
`legacy_relative_z`, 작업거리, Z 상승량은 사용하지 않는다.

```bash
.venv/bin/python src/tools/test_platform_motion.py --port /dev/ttyACM0
.venv/bin/python src/tools/test_platform_motion.py --port /dev/ttyACM0 --z <user-safe-z> --execute
.venv/bin/python src/tools/test_platform_motion.py --port /dev/ttyACM0 --roll <user-small-angle> --execute --ack-safe-height
.venv/bin/python src/tools/test_platform_motion.py --port /dev/ttyACM0 --pose-json <path> --dry-run
```

`--log result.json` 또는 `--log result.csv`로 before/during/after telemetry를
저장할 수 있다. timeout 시 추가 motion이나 존재하지 않는 STOP/ESTOP/ABORT를
전송하지 않는다. 실제 비상 대응은 사용자 감독, SMPS/전원 차단, firmware limit에
의존한다.

### Command 이후 stable 판정

USB/CDC와 host serial RX에는 물리적 이동이 끝난 뒤에도 과거 telemetry가 남을 수 있다.
따라서 read-only snapshot은 `reset_input_buffer() → --fresh-settle 대기 →
reset_input_buffer() → valid TLM 수신` 순서의 `read_fresh_telemetry()`를 사용한다.
settle 중 뒤늦게 host에 도착한 packet까지 두 번째 reset에서 제거한다. final reset 이후
malformed TLM은 건너뛰고 다음 valid TLM을 반환하지만, 기존 strict parser 자체와 일반
`read_telemetry()`의 malformed 실패 동작은 변경하지 않는다.

motion diagnostic은 각 command 직전 reset 정책을 유지한다. write/flush 직후에는 같은
이중 fresh RX 경계를 다시 만든 뒤에만 stable 판정을 시작한다. 따라서 command 이전이나
write 중 이미 host RX에 queued된 telemetry는 완료 판정에 참여하지 않는다. Automatic Z의
각 candidate 이동도 `PlatformMotionDiagnostic`을 사용하므로 동일한 정책을 상속한다.

완료 판정은 첫 `stable=True` 표본 하나를 사용하지 않는다.

1. `--post-command-guard` 구간의 telemetry는 기록만 하고 완료 판정에서 제외한다.
2. guard 이후 `stable=False`를 관측하면 실제 motion/settling이 시작된 것으로 본다.
3. `--stable-samples`개의 연속된 `stable=True`를 확인해야 완료한다.
4. firmware deadband 때문에 `stable=False`가 없으면 `--deadband-observation` 시간이
   지난 뒤 연속 stable 조건으로 완료할 수 있다.

기본값 `guard 0.05초 / stable 3개 / deadband 관찰 0.20초 / fresh settle 0.10초`는
calibration되지 않은 diagnostic timing 설정이며 production safety limit가 아니다.
장비 telemetry 주기와 USB drain 관찰 결과에 맞춰 CLI의 `--post-command-guard`,
`--stable-samples`, `--deadband-observation`, `--fresh-settle`로 변경할 수 있다.

이 경계가 보장하는 것은 "마지막 host RX reset 이후 수신"이다. 현재 firmware TLM에는
device sequence, device timestamp, command ACK가 없으므로 packet이 STM에서 command 이후
생성되었다는 사실까지 host만으로 절대 증명할 수는 없다. 그 수준의 보장이 필요하면
firmware protocol에 sequence/timestamp 또는 command ACK를 추가해야 한다. 또한
`SerialPlatformController.wait_until_stable()` 단독 호출은 command 경계를 만들지 않으므로
실제 command 완료 판정에는 사용하지 않는다.

```bash
.venv/bin/python src/tools/test_platform_motion.py \
  --port /dev/ttyACM0 --snapshot --fresh-settle 0.10

.venv/bin/python src/tools/read_platform_telemetry.py \
  --port /dev/ttyACM0 --snapshot --fresh-settle 0.10
```

`read_platform_telemetry.py --count N`도 첫 packet은 fresh-read로 얻고, 나머지는 그 경계
이후의 연속 stream을 읽는다. `read_fresh_telemetry(timeout)`의 timeout은 drain 이후
valid TLM 대기에 적용되므로 최대 호출 시간은 대략 `fresh-settle + timeout`이다.

실제 motion은 verified limit와 safe Z가 아직 없으므로 system hardware factory에 연결하지 않았다.

## 독립 부분 통합 hardware cycle

`src/tools/test_integrated_inspection_cycle.py`는 production `run_system.py --mode hardware`를
활성화하지 않고 다음 범위만 실제 연결하는 명시적 diagnostic이다.

```text
Projector BLACK
→ Conveyor F<user steps> / Target Reached
→ Structured Light 4-phase / BLACK
→ current-run pose JSON / dominant Roll·Pitch
→ user safe Z / stable
→ Roll·Pitch / stable
→ exact LED_ON checkpoint
→ highest-passing Automatic Z / best Z return
→ BLACK / 종료
```

Conveyor steps, safe Z, Z candidates, z_max는 모두 CLI 필수 입력이며 코드에 장비값을
숨기지 않는다. `--execute`, `--ack-mechanical-range`, 정확한 `EXECUTE` 입력이 모두
있어야 projector GUI, serial, camera 또는 structured-light subprocess를 연다. 기본 실행은
정적 설정만 검증하는 DRY RUN이다. Roll/Pitch는 이번 scan의
`FINAL_DC_MASK_PHASE*_pose.json`에서 읽으며 `legacy_relative_z`는 사용하지 않는다.
`RealPosePlanner`의 production motion 차단 metadata도 변경하지 않는다.
모든 platform 이동의 공통 telemetry/stable 대기 상한은 `--platform-motion-timeout`으로
지정하며 기본값은 30초인 diagnostic timeout이다. 이는 motion speed, Z target 또는
completion 판정을 변경하지 않고, command 이후 stable telemetry를 기다릴 수 있는 시간만
늘린다.

Structured-light launcher는 요청한 monitor 이름을 child capture에 전달한다. runner는 scan
전후의 `촬영_*` 및 pose file snapshot을 비교하여 이번 invocation에서 새로 생성되거나 갱신된
directory와 pose JSON만 허용한다. 같은 초의 기존 directory가 갱신되더라도 변경되지 않은 과거
pose는 거부한다. Scan shell은 timestamped integration run의 `structured_light/raw`로 직접
출력하며 timeout/Ctrl+C에는 새 session의 process group 전체를 종료해 camera/GUI descendant를
남기지 않는다. Conveyor도 command 직전에
`reset_input_buffer → configurable settle → reset_input_buffer`로 과거 completion marker를
버리고, 현재 direction/steps echo 이후의 completion만 인정한다.

LED checkpoint 이전에는 Orbbec을 시작하지 않는다. Automatic Z는 기존
`SurfaceReadinessEvaluator`의 80–2000 mm, depth/plane/patch/FOV, 8-frame readiness를
그대로 사용하며 selection policy는 `highest_passing_readiness`로 고정한다. 각 후보는
`BLACK 확인 → Z 이동/stable → BLACK 재확인 → RGB+Depth capture` 순서다. anomaly,
heatmap, NORMAL/DEFECT, conveyor OUT, LED serial control은 호출 경로 자체에 없다.

실행 결과는 `results/integrated_hardware/run_YYYYMMDD_HHMMSS/` 아래에 저장한다.

```text
cycle_result.json
structured_light/  # raw current scan, run_info, manifest/pose snapshot, relative current_run link
automatic_z/       # candidate RGB/depth, result JSON/CSV, quality config snapshot
telemetry/         # shared platform JSON/CSV
logs/              # stage 및 structured-light stdout/stderr
```

실패 시 실패 stage/error를 보존하고 BLACK 복귀와 resource close만 시도한다. 임의 복귀
motion, STOP/ESTOP/ABORT, conveyor OUT은 보내지 않는다. `projector_final_state`는 close 직전
BLACK 복구 결과이며 `projector_state_after_close`는 window destroy 이후 상태다. 현재
OpenCV window는 process-owned이므로 CLI 종료 뒤에도 HDMI desktop을 영구 차단하는 것은
보장하지 않는다.

현재 legacy 구조광 capture는 parent의 persistent BLACK window와 별도의 fullscreen phase
window를 child process에서 연다. 따라서 첫 실장 실행에서는 HDMI-0에서 실제 phase window가
BLACK window 위에 표시되는지, `000 → 090 → 180 → 270` 이후 parent BLACK이 즉시 다시
보이는지 눈으로 확인해야 한다. 또한 scan-derived Roll/Pitch에는 아직 calibrated host numeric
limit가 없으므로 `--ack-mechanical-range` 전에 이번 장비에서 검증된 범위인지 운영자가
확인해야 한다.

DRY RUN:

```bash
.venv/bin/python src/tools/test_integrated_inspection_cycle.py \
  --conveyor-port /dev/ttyUSB0 \
  --platform-port /dev/ttyACM0 \
  --lighting-port /dev/ttyACM1 \
  --conveyor-steps 6325 \
  --monitor HDMI-0 \
  --safe-z 20 \
  --z-start 20 \
  --z-max 30 \
  --z-coarse-step 5 \
  --z-fine-step 1 \
  --pose-plan-mode all_valid_planes \
  --quality-config config/automatic_z_quality.json
```

실제 실행은 같은 명령에 다음 두 flag를 추가하고, 이후 terminal prompt에 `EXECUTE`,
자세 준비 후 checkpoint에 `LED_ON`을 정확히 입력한다.

```text
--execute --ack-mechanical-range
```

## Uno inspection lighting boundary

검증된 `sketch_aug20a/sketch_aug20a.ino` contract를 사용한다. Uno는 9600 baud로 동작하며
startup에서 `NeoPixel controller started.`와 `LEDs are initially OFF.`를 출력한다. inspection
조명 ON은 기존 도장면용 Neutral White 명령 `2`와 ACK `Mode: Neutral White`를 사용하고,
OFF는 `0`과 ACK `Mode: All LEDs OFF`를 사용한다. 새로운 `LED:ON` 또는 brightness command는
생성하지 않는다. startup banner를 모두 읽지 못하거나 ACK timeout이 발생하면 cycle은 실패한다.

Integrated cycle의 lighting 경계는 다음과 같다.

```text
PLATFORM_CONNECT → LIGHTING_CONNECT → LIGHTING_OFF → PROJECTOR_BLACK
→ STRUCTURED_LIGHT_SCAN (LED OFF)
→ LIGHTING_ON → plane safe-Z/R-P/Automatic-Z loop
→ READY_FOR_ANOMALY per plane
→ PROJECTOR_BLACK → LIGHTING_OFF → cleanup
```

실제 CLI에서는 `--lighting-port`와 `--lighting-startup-timeout`으로 port 및 startup timeout을
주입한다. `inspection_led_initial_off`, `inspection_led_on`, `inspection_led_off_at_end`,
`lighting_error`와 lighting stage가 `cycle_result.json`에 저장된다. 현재 anomaly inference는
호출하지 않으며, plane result는 `ready_for_anomaly=true`, `anomaly_executed=false`로 남는다.

## Orbbec camera 경계

`OrbbecCameraController`는 기존 Gemini 336L profile/config/aligned RGB+Depth helper를 lifecycle API(`start/capture/close`)로 감싼다. 결과는 `RGBDepthFrame(color_bgr, depth_mm, timestamp)`이다. 실제 surface 계산은 계속 `src.core`가 담당한다.

```bash
python3 src/tools/test_orbbec_camera.py --output-dir /tmp/orbbec_test
```

이 명령은 사용자가 직접 실행할 때만 camera를 연다. 현재 controller는 검증된 helper를 재사용하므로, 다음 단계에서 helper를 experiment script 밖의 camera module로 완전히 추출해야 한다.

## Structured Light shell 경계

`서영 파트 파일`의 구조광 알고리즘은 보존하고 경로 및 실행 경계만 portable하게 관리한다. `ShellStructuredLightRunner`는 subsystem root, result root, interpreter, timeout, non-interactive 및 visualization 정책을 config로 받는다.

```bash
.venv/bin/python src/tools/check_structured_light.py \
  --subsystem-root "$PWD/서영 파트 파일" \
  --result-root "$PWD/서영 파트 파일/플랫폼 바닥 따기/구조광_전처리/샘플"
```

기본은 preflight only다. 실제 실행에는 `--execute`가 명시적으로 필요하다. 소스와 환경이 준비됐지만 Dongjin 배치의 보정 파일이 아직 없으면 `CALIBRATION_REQUIRED`를 반환한다. 이것은 path/source failure가 아니다.

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

Automatic Z는 `--z-candidates`를 사용하는 기존 explicit mode와,
`--z-start`, `--z-max`, `--z-coarse-step`, `--z-fine-step`를 사용하는 adaptive mode를
지원한다. Adaptive mode는 첫 coarse FAIL과 직전 PASS 사이만 fine step으로 검사하며,
각 후보의 stable telemetry와 RGB/depth readiness 판정은 기존 방식 그대로 유지한다.

`--pose-plan-mode all_valid_planes`를 선택하면 초기 structured-light scan 1회 후 pose JSON의
모든 유효 plane을 dominant 우선, points count 내림차순으로 순차 검사한다. 각 plane은
`safe_z stable → Roll/Pitch stable → Automatic Z` 순서이며, 다음 plane 전에도 `safe_z stable`을
거친다. 결과는 `plane_00/`, `plane_01/` 아래에 분리 저장되고 현재 AE/anomaly 단계는
실행하지 않으며 `ready_for_anomaly=true`로만 보존한다.

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
