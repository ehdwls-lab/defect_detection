# defect_detection — Codex Working Rules

## Goal

Graduation/Hanium project:
Active Vision + Structured Light based automated surface defect inspection.

Current priority:
1. Stabilize integrated hardware inspection.
2. Collect production-condition normal GRAY data.
3. Train GRAY_v1 anomaly-detection autoencoder.
4. Optimize execution time only after functional validation.

Keep changes minimal and evidence-based.


## Critical rule

Before modifying code:
- Read the actual current implementation and relevant tests.
- Reuse existing helpers/configs.
- Do not redesign working subsystems without evidence.
- Do not fix unrelated problems.
- Do not run hardware unless the user explicitly asks for a hardware run.


## Frozen subsystems

Do NOT modify unless explicitly requested:

- Structured Light algorithm
- camera ↔ platform calibration K
- plane extraction / multi-plane logic
- board-plane selection
- ArUco fallback
- signed-height ROI
- Depth voting
- contour fill
- RGB-assisted ROI fallback
- ROI erosion = 10 px
- patch size = 64
- patch stride = 32
- anomaly surface coverage = 1.0
- patchable ratio threshold = 0.5
- anomaly preprocessing
- AE architecture
- GRAY_v1 production-mask dataset pipeline
- Auto-Z depth readiness = 0.25
- Auto-Z quality weights:
  - surface = 0.6
  - depth-valid = 0.4


## Platform hardware limits

Mechanical limit:
- Roll = ±30 deg
- Pitch = ±30 deg

Inspection operational axis limit:
- Roll = ±25 deg
- Pitch = ±25 deg

Do NOT change the mechanical ±30 deg limit.


## Adaptive R/P/Z inspection motion

The platform is below the conveyor and lifts the specimen upward.

Required production motion concept:

1. Keep platform level: R=0, P=0.
2. Raise to tilt-entry Z=25.
3. Apply the safe inspection R/P.
4. Search downward:
   Z=25 → 24 → 23 → 22 → 21 → 20 → 19 → 18 → 17.
5. Adapt R/P to the Z-dependent combined-tilt safety envelope.
6. Perform final geometry / ROI / RGB / anomaly.
7. Before lowering for cleanup:
   - raise safely to Z=25,
   - return R=0, P=0,
   - then lower to cleanup/safe Z.

Never lower a strongly tilted platform directly toward the conveyor.


## Z-dependent combined-tilt envelope

Maximum combined tilt:

- Z17 → 20 deg
- Z18 → 21 deg
- Z19 → 22 deg
- Z20 → 23 deg
- Z21 and above → 25 deg

For fractional Z, use the existing interpolation implementation.

Combined tilt must be based on platform orientation, not abs(R)+abs(P).

When the requested R/P exceeds the allowed combined tilt:
- preserve signs,
- preserve R:P direction/ratio,
- scale both together.

Do not terminate the Z search merely because a lower Z requires a smaller tilt.


## Safe transition ordering

Descending to a lower Z:
1. Compute R/P allowed at next Z.
2. Reduce R/P first while still at current Z.
3. Confirm motion.
4. Move Z downward.

Ascending to a higher Z:
1. Keep current safe R/P.
2. Raise Z first.
3. Apply increased R/P only after reaching the higher Z.

Reuse the project safe-motion helper. Do not duplicate raw motion logic.


## Projector cover mapping

Physical hardware mapping is fixed:

- OPEN = 90 deg
- CLOSE = 0 deg

Production CLI must use:

--cover-open-angle 90
--cover-close-angle 0
--cover-cleanup-state CLOSE

Never reverse these values.


## Serial ports

Conveyor Mega:
 /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0

Platform STM32:
 /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066FFF383133524157152339-if02

Lighting / cover Arduino:
 /dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_75932313039351C09122-if00

All use 115200 baud.

Do not replace these with /dev/ttyUSB0 placeholders in production commands.


## Camera / final inspection conditions

Orbbec Gemini 336L.

Final RGB inspection:
- Color AE = false
- Color AWB = false
- Brightness = 48
- Exposure = 1100
- Gain = 64
- White balance = 4600

Depth:
- AE = false
- Exposure = 3000
- Gain = 16

Optical sequence:
1. Structured Light normally.
2. Projector BLACK.
3. Physical projector cover CLOSE.
4. Final Geometry with LED OFF.
5. Neutral White LED ON.
6. ROI Depth acquisition.
7. Final RGB.
8. Anomaly inference.
9. LED OFF cleanup.

Do not silently change this sequence.


## Training / GRAY_v1

Production training source:

final_capture/final_rgb.png
+
anomaly/inspection_mask.png

Training patch rule:
- 64x64
- stride 32
- mask coverage 1.0

PatchDataset preprocessing order:
full image preprocessing
→ RGB conversion
→ crop patch

Do not export preprocessed patch PNGs and preprocess them again.

GRAY_01:
- train / validation source

GRAY_02 and GRAY_03:
- hold-out TEST only
- never include in GRAY_v1 train/val

Split train/val by run or independent pose source.
Do not randomly split overlapping patches from the same image into train and validation.

Visible scratch/dent patches must not be included as normal training samples.


## Python invocation

Because src/platform can shadow Python stdlib platform when scripts are invoked directly:

Use:
 python -m src.train
 python -m src.infer_anomaly

Do NOT use:
 python src/train.py
 python src/infer_anomaly.py


## Testing after code changes

Run only focused tests while developing.

After implementation is stable:

python -m compileall -q src
python -m unittest <focused test modules>
python -m unittest discover -s tests -p 'test*.py'
git diff --check

Do not repeatedly run the full suite after every small edit.


## Token / context efficiency

Be concise.

- Do not dump entire large files unless required.
- Search for relevant symbols first.
- Read only relevant line ranges.
- Do not repeatedly re-read files already inspected unless changed context requires it.
- Do not paste full unittest output when tests pass.
- Report successful tests as counts, e.g. "285 passed".
- On failure, inspect only the failing traceback/test.
- Avoid printing huge hardware logs.
- Use grep / tail / focused extraction when possible.
- Do not run web searches for repository coding tasks unless explicitly requested.
- Do not explain every tool call.
- Final report should contain only:
  1. files changed,
  2. behavior changed,
  3. tests,
  4. unresolved issues,
  5. GO / NO-GO when relevant.


## Git

- Do not commit unless explicitly requested.
- Do not create branches unless explicitly requested.
- Do not reset or discard unrelated user changes.
- Keep diffs minimal.
