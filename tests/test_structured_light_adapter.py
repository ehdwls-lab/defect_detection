from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.integration.coordinate_contract import CloudType
from src.integration.structured_light_adapter import StructuredLightAdapter, UnsupportedPLYFormatError


HEADER = """ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
0 0 0 1 2 3 0 0 1
"""


class StructuredLightAdapterTests(unittest.TestCase):
    def test_header_and_artifact_selection(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            names = (
                "03_v2_현재프레임기준_최종_물체만.ply",
                "02_v2_현재프레임기준_최종_물체+플랫폼.ply",
                "FINAL_DC_MASK_PHASE_z30_SIGN_PLUS_WITH_FLOOR.ply",
                "mystery.ply",
            )
            for name in names:
                (root / name).write_text(HEADER, encoding="ascii")
            artifacts = StructuredLightAdapter.discover_artifacts(root)
            mapping = {path.name: kind for path, kind in artifacts}
            self.assertIs(mapping[names[0]], CloudType.OBJECT_ONLY)
            self.assertIs(mapping[names[1]], CloudType.OBJECT_AND_PLATFORM)
            self.assertIs(mapping[names[2]], CloudType.OBJECT_PLATFORM_FLOOR)
            self.assertIs(mapping[names[3]], CloudType.UNKNOWN)
            result = StructuredLightAdapter.from_directory(root)
            self.assertEqual(result.ply_path.name, names[0])
            self.assertTrue(result.cloud.has_normals)
            self.assertEqual(len(result.metadata["available_artifacts"]), 4)

    def test_binary_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            ply = Path(raw) / "object_only.ply"
            ply.write_bytes(HEADER.replace("format ascii 1.0", "format binary_little_endian 1.0").encode("ascii"))
            with self.assertRaises(UnsupportedPLYFormatError):
                StructuredLightAdapter.parse_ply_header(ply)


if __name__ == "__main__":
    unittest.main()
