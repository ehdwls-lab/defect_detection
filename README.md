# defect_detection

Active Vision + Structured Light 기반 표면 결함 자동 검사 프로젝트입니다.

## Repository layout

- `src/`: 검사, 카메라, 이상 탐지, 하드웨어 통합 코드
- `tests/`: 하드웨어를 열지 않는 자동 테스트
- `config/`: 운영 설정
- `docs/`: 통합 규약과 운영 문서
- `서영 파트 파일/`: 구조광 소스와 현재 배치 보정 데이터
- `3dof_PID_Select/`, `nema34test/`, `sketch_aug20a/`: 펌웨어
- `archive/`: 운영 경로에서 분리한 레거시/정리 대상

`data/`, `results/`, `Log/`, `.venv/`, 모델 가중치는 Git에 포함하지 않습니다. 과거 구조광 촬영 산출물은 `archive/pending_cleanup/structured_light_samples/`에 보관했으며, 확인 후 삭제할 수 있습니다.

## Setup on another laptop

Python 3.10 기준입니다.

```bash
git clone https://github.com/ehdwls-lab/defect_detection.git
cd defect_detection
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

실제 카메라/구조광 환경은 추가로 설치합니다.

```bash
python -m pip install -r requirements-hardware.txt
```

관찰용 UI가 필요하면:

```bash
python -m pip install -r requirements-ui.txt
```

## Verification

다음 명령은 실제 하드웨어를 열지 않습니다.

```bash
python -m compileall -q src
python -m unittest discover -s tests -p 'test*.py'
```

학습과 이상 탐지 CLI는 패키지 모드로 실행합니다.

```bash
python -m src.train --help
python -m src.infer_anomaly --help
```

실제 하드웨어 실행은 장비별 시리얼 권한, Orbbec SDK, 프로젝터 모니터 배치, 현재 배치 보정값을 확인한 뒤에만 진행하세요. 세부 계약은 `docs/`를 참고하세요.
