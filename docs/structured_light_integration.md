# Structured Light Integration Interface

## 1. Goal

This document defines the integration boundary between the external structured-light subsystem and the defect-detection runtime. The goal is to keep the structured-light pipeline intact while exposing a small, stable data contract to the later stages:

Structured Light subsystem
  -> Integration Adapter
  -> Pose Planner
  -> Platform Controller
  -> Automatic Z Search
  -> Surface-only defect detection

The structured-light subsystem remains the source of geometry generation. It is not modified by the defect-detection repo.

---

## 2. Role separation

### Structured Light subsystem

Responsibilities:

- capture low-Z structured-light data
- recover phase / depth / mask
- generate PLY or point-cloud outputs
- create object mask and platform/floor artifacts

The geometry algorithms remain preserved in `서영 파트 파일`. Integration changes are limited to
portable paths, launch policy, preflight, and machine-readable metadata.

## Phase 4A portable operation

Canonical subsystem root:

```text
/home/dongjin/defect_detection/서영 파트 파일
```

The scripts use their own directory by default. `STRUCTURED_LIGHT_ROOT` may override it.
Python selection order is `STRUCTURED_LIGHT_PYTHON`, the repository `.venv/bin/python`, then
`python3`. Calibration is deliberately manual and separate from `SystemController`:

```bash
bash "서영 파트 파일/초기세팅.sh"
```

This creates, without fake defaults:

```text
프로젝터 수동 범위 확인/프로젝터_세로범위.json
플랫폼 바닥 따기/현재배치_기준데이터/active/E1999_G64/플랫폼_바닥_depth.npy
```

Filesystem-only preflight:

```bash
.venv/bin/python src/tools/check_structured_light.py \
  --subsystem-root "$PWD/서영 파트 파일" \
  --result-root "$PWD/서영 파트 파일/플랫폼 바닥 따기/구조광_전처리/샘플"
```

Before calibration the expected status is `CALIBRATION_REQUIRED`, with
`source_ready=true` and `environment_ready=true`. Inspection defaults to non-interactive mode
when launched through `ShellStructuredLightRunner`; manual shell execution retains the placement
prompt. CloudCompare is off unless `STRUCTURED_LIGHT_VISUALIZE=1` is set.

## Projector-only diagnostic

Before changing exposure or phase thresholds, validate the PC-to-projector path without opening
the Orbbec camera:

```bash
.venv/bin/python src/tools/test_structured_light_projector.py
```

Keys `1/2/3` show black, white, and 50% gray; `4/5/6/7` show the exact production
0/90/180/270 phase arrays; `8` repeats the production sequence; `9` shows the coverage screen.
Press `Q` or `ESC` to quit. The four production arrays are also saved under
`/tmp/projector_test`. GUI-free numeric verification is available with `--array-only`.

The shared generator preserves `direction=horizontal`, `period=80`, `base=128`, and
`amplitude=127`. In the inherited naming convention, `horizontal` varies by image row and
therefore produces horizontal bands. Do not change this direction merely to match a visual label.

### Integration Adapter

Responsibilities:

- locate the latest structured-light result
- validate the result directory or PLY path
- normalize file paths and metadata
- classify cloud type
- attach coordinate semantics and cloud metadata
- return a stable `StructuredLightResult`

The adapter must not:

- re-implement phase unwrapping
- generate new PLY files
- perform anomaly detection
- move the platform

### Pose Planner

Responsibilities:

- consume a `StructuredLightResult`
- decide which surface or object region is inspectable
- estimate pitch / roll target(s)
- produce a `PoseTarget` or `InspectionPlan`

This layer is separate from the platform controller and from the final anomaly detector.

### Platform Controller

Responsibilities:

- receive `PoseTarget`
- issue safe move commands to the actuator / STM layer
- coordinate a safe platform sequence

### Automatic Z Search

Responsibilities:

- refine the inspection Z after the platform is pitched or rolled
- use RGB + depth quality heuristics
- choose the best inspection focus plane

### Anomaly Detection

Responsibilities:

- inspect the final RGB + depth frame
- generate surface-only patches
- apply autoencoder scoring
- classify NORMAL / DEFECT

---

## 3. Supported input modes

The adapter supports two entry points:

1. `StructuredLightAdapter.from_directory(path)`
   - used when a structured-light run directory exists
   - the adapter locates the final PLY, masks, phase, depth, and metadata

2. `StructuredLightAdapter.from_ply(ply_path)`
   - used when a final object cloud is already available
   - the adapter validates the PLY and produces metadata without re-segmenting the cloud

The adapter does not invent new naming conventions. It reads the real structured-light outputs and normalizes them.

---

## 4. Core data contract

### Cloud type contract

The adapter must explicitly label the point cloud type. Valid categories:

- `OBJECT_ONLY`
- `OBJECT_AND_PLATFORM`
- `OBJECT_PLATFORM_FLOOR`
- `UNKNOWN`

This is required because the downstream pose planner may require object-only geometry while the final visualization may include floor points.

If a required cloud does not exist, the contract returns a clear error such as:

- `PLYNotFoundError`
- `UnsupportedCloudTypeError`
- `ObjectCloudUnavailableError`

The adapter must not silently fall back to an arbitrary point set.

### Coordinate convention

The coordinate contract is mandatory. Each result must carry the meaning of X, Y, and Z. For the current structured-light analysis, the following interpretation is the safest contract:

- X: image-centered relative horizontal coordinate, derived from pixel column and image center offset
- Y: image-centered relative vertical coordinate, derived from pixel row and image center offset
- Z: relative phase-derived height, not a calibrated absolute mm distance unless proven elsewhere
- Units: X/Y are in the existing PLY convention; Z is a relative phase-based height scale

This is intentionally conservative. We do not guess a metric conversion unless the structured-light code proves it.

### Coordinate metadata

Each `StructuredLightResult` should carry:

- x-axis meaning
- y-axis meaning
- z-axis meaning
- xy unit
- z unit
- origin description
- image width and height when available

---

## 5. Recommended dataclasses

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CloudType(str, Enum):
    OBJECT_ONLY = "OBJECT_ONLY"
    OBJECT_AND_PLATFORM = "OBJECT_AND_PLATFORM"
    OBJECT_PLATFORM_FLOOR = "OBJECT_PLATFORM_FLOOR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CoordinateConvention:
    x_axis: str
    y_axis: str
    z_axis: str
    xy_unit: str
    z_unit: str
    origin_description: str
    image_width: int | None = None
    image_height: int | None = None


@dataclass(frozen=True)
class PointCloudMetadata:
    ply_path: Path
    point_count: int
    has_color: bool
    has_normals: bool
    includes_object: bool
    includes_platform: bool
    includes_floor: bool
    cloud_type: CloudType
    coordinate: CoordinateConvention


@dataclass(frozen=True)
class StructuredLightResult:
    run_id: str
    ply_path: Path
    object_mask_path: Path | None
    phase_path: Path | None
    depth_path: Path | None
    cloud: PointCloudMetadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseTarget:
    pose_id: str
    pitch_deg: float | None = None
    roll_deg: float | None = None
    target_surface_id: str | None = None
    confidence: float | None = None
    source: str = "pose_planner"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectionPlan:
    object_id: str
    poses: list[PoseTarget]
    source_ply: Path
    metadata: dict[str, Any] = field(default_factory=dict)
```

These structures are intentionally simple and may be adjusted to match repo conventions, but the contract must remain explicit.

---

## 6. PLY contract

The adapter must classify which PLY it is handling. The object cloud may come in several variants:

- object only
- object + platform
- object + platform + floor

A downstream module must know which variant it is receiving. The adapter must not reinterpret or remove points silently.

It is acceptable for the adapter to expose the following helpers:

- `is_object_only(result) -> bool`
- `requires_pose_from_object_cloud(result) -> bool`
- `get_required_cloud_type(required_kind: CloudType) -> StructuredLightResult`

If a required cloud type does not exist, the adapter should raise a specific error rather than choosing a different cloud by guesswork.

---

## 7. Pose planner contract

The raw structured-light result should not be passed directly to the platform controller. It must first flow through a pose-planning layer.

```python
@dataclass(frozen=True)
class PosePlanningInput:
    cloud: PointCloudMetadata
    ply_path: Path
    object_mask_path: Path | None
    coordinate: CoordinateConvention
    metadata: dict[str, Any] = field(default_factory=dict)
```

The pose planner is allowed to:

- read the object cloud
- estimate a candidate plane or surface
- compute a target pitch / roll
- return one or more `PoseTarget` candidates

The pose planner is not allowed to:

- issue final platform commands
- compute final inspection Z values on its own
- invent new geometry rules

---

## 8. Z policy

The pose planner is responsible for pitch and roll only.

The final inspection Z is not selected in the pose-planning stage. It is selected later in the camera-based automatic Z search process.

This distinction matters:

- `PoseTarget.pitch_deg` and `PoseTarget.roll_deg` define viewing orientation
- `inspection_z` is determined later using RGB + depth / quality criteria

The adapter and pose planner should therefore avoid mixing a structural height estimate with the final machine focus position.

---

## 9. Platform command separation

The platform controller sits after the pose planner:

`PoseTarget -> PlatformController -> STM command`

This keeps the geometry layer and the machine-control layer separate.

The adapter and pose planner must not produce raw STM strings such as `Z:<cm> R:<deg> P:<deg>`. Those are platform-specific and should be assembled only in the controller layer.

---

## 10. Filesystem strategy

The structured-light code currently contains absolute paths in its external environment. The defect-detection repo should not embed those paths directly. Instead, the adapter should use a path container such as:

```python
@dataclass(frozen=True)
class StructuredLightPaths:
    root: Path
    initial_setup_dir: Path | None = None
    current_run_dir: Path | None = None
```

This allows environment-specific injection without writing hardcoded `/home/...` values into the defect-detection codebase.

---

## 11. Metadata manifest strategy

Even when the external structured-light code is not changed, it is useful to store a manifest in the output directory. This manifest can be created by the adapter rather than by the external code.

Example:

```json
{
  "schema_version": "structured_light_interface_v1",
  "run_id": "...",
  "ply": {
    "path": "...",
    "type": "OBJECT_ONLY",
    "point_count": 123456
  },
  "coordinate": {
    "x": "image-centered horizontal axis",
    "y": "image-centered vertical axis",
    "z": "relative phase-derived height",
    "xy_unit": "pixel-relative",
    "z_unit": "relative_phase_scale",
    "origin": "image center"
  },
  "artifacts": {
    "object_mask": "...",
    "phase": "...",
    "depth": "..."
  }
}
```

This keeps the adapter compatible with future changes in the external subsystem.

---

## 12. Error contract

The adapter and planner should raise explicit exceptions rather than generic `Exception` values. Recommended categories:

- `StructuredLightLoadError`
- `PLYNotFoundError`
- `PLYReadError`
- `UnsupportedCloudTypeError`
- `ObjectCloudUnavailableError`
- `CoordinateMetadataMissingError`
- `PosePlannerInputError`

These allow the calling layer to decide whether to stop execution, warn the operator, or retry.

---

## 13. Versioning

The integration contract includes a schema version to avoid breaking downstream code when the external pipeline changes.

Recommended version pattern:

- `structured_light_interface_v1`
- later revisions can extend the metadata while preserving compatibility

---

## 14. Integration sequence

The intended flow is:

```python
structured_result = StructuredLightAdapter.from_directory(run_dir)
inspection_plan = PosePlanner.plan(structured_result)

for pose in inspection_plan.poses:
    platform.move_to_safe_pose(pose)
    best_z = automatic_z_search(...)
    inspection_result = anomaly_detector.inspect(...)
```

This is the contract the code should express; the actual STM and AE logic remain out of scope for this phase.

---

## 15. Out-of-scope for this step

The current phase does not include:

- modifying the structured-light algorithm
- recreating PLY generation logic
- writing raw STM commands
- implementing autoencoder inference
- computing final inspection Z
- adding arbitrary geometric transforms
- implementing surface-normal estimation without a proven source

This stage is intentionally limited to a stable integration interface.

---

## 16. Recommended repo layout

The integration layer should live in a separate module section, such as:

- `src/integration/structured_light_adapter.py`
- `src/integration/pose_target.py`
- `src/integration/inspection_plan.py`
- `src/integration/pose_planner.py`

The repo should keep structured-light geometry generation outside this package and treat the integration layer as a strict parser / validator / contract layer.

---

## 17. Implementation expectation

The next implementation step should be small and safe:

1. create dataclasses for the contract
2. implement PLY path discovery and validation
3. expose cloud-type / metadata parsing
4. add a placeholder pose planner interface
5. validate the adapter with representative structured-light outputs

Not yet included:

- pitch / roll algorithm implementation
- automatic Z search
- anomaly detector integration
- platform movement logic

---

## 18. Summary

The fundamental rule is simple:

The structured-light subsystem produces the geometry.
The adapter normalizes it.
The pose planner selects inspection orientation.
The platform controller executes motion.
The anomaly detector performs the final defect classification.

This keeps the boundary clear and prevents the structured-light pipeline and the defect-detection pipeline from becoming tightly coupled before the data contract is stable.
