from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SOURCE = Path(__file__).parents[1] / "서영 파트 파일" / (
    "구조광_전처리_최종_v2_현재프레임플랫폼기준_Depth홀위상보강_경로수정_0822 (1).py"
)


def load_mask_function():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "현재프레임_마스크저장",
        "현재프레임_가장큰연결영역",
        "현재프레임_마스크팽창",
        "현재프레임_플랫폼피팅마스크_생성",
    }
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "np": np, "cv2": cv2, "Path": Path, "json": json,
        "현재프레임_물체제외_팽창PX": 15,
        "현재프레임_최소플랫폼픽셀": 2000,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["현재프레임_플랫폼피팅마스크_생성"]


class CurrentFramePlatformDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_do_not_change_mask_and_save_all_stage_artifacts(self):
        function = load_mask_function()
        domain = np.zeros((80, 100), dtype=bool)
        domain[5:75, 5:95] = True
        object_area = np.zeros_like(domain)
        object_area[2:78, 2:98] = True
        object_mask = np.zeros_like(domain)
        object_mask[30:50, 42:58] = True

        expected = function(object_mask, domain, object_area)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            counts = {
                "object_area_count": int(object_area.sum()),
                "domain_count": int(domain.sum()),
            }
            actual = function(
                object_mask, domain, object_area,
                diagnostics={"directory": directory, "counts": counts, "preview": None},
            )
            np.testing.assert_array_equal(actual, expected)
            for name in (
                "06_object_excluded.png", "07_area_eroded.png",
                "08_platform_candidate.png", "09_largest_component.png",
                "10_platform_fit.png", "platform_mask_diagnostics.json",
            ):
                self.assertTrue((directory / name).is_file(), name)
            payload = json.loads((directory / "platform_mask_diagnostics.json").read_text())
            self.assertEqual(payload["final_platform_fit_count"], int(actual.sum()))
            self.assertEqual(payload["minimum_required"], 2000)
            self.assertGreaterEqual(payload["connected_component_count"], 1)


if __name__ == "__main__":
    unittest.main()
