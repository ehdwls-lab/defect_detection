from __future__ import annotations

import unittest

from src.inspection.mock_quality_sampler import MockQualitySampler
from src.system.factory import build_system
from src.system.states import SystemState


class SystemMockWorkflowTests(unittest.TestCase):
    def test_two_pose_workflow(self):
        controller = build_system("mock")
        result = controller.run_once()
        self.assertTrue(result.success)
        self.assertTrue(result.mock)
        self.assertEqual(result.final_status, "MOCK_COMPLETE")
        self.assertEqual(len(result.pose_results), 2)
        self.assertTrue(all(item.best_z_cm == 21.5 for item in result.pose_results))
        self.assertTrue(all(item.anomaly_result["status"] == "MOCK_NORMAL" for item in result.pose_results))
        self.assertEqual(result.state_history[0], SystemState.INITIALIZING.value)
        self.assertEqual(result.state_history[-1], SystemState.STOPPED.value)
        self.assertIn(SystemState.READY.value, result.state_history)
        self.assertIn(SystemState.CONVEYOR_TO_INSPECTION.value, result.state_history)
        self.assertNotIn(SystemState.WAIT_OBJECT.value, result.state_history)
        self.assertIn("mock moved out", controller.conveyor.events)

    def test_no_valid_z_is_structured_failure(self):
        controller = build_system("mock")
        controller.quality_sampler = MockQualitySampler(all_fail=True)
        result = controller.run_once()
        self.assertFalse(result.success)
        self.assertEqual(result.failed_state, SystemState.AUTO_Z_SEARCH.value)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("NoValidInspectionZ", result.error_message)
        self.assertEqual(result.state_history[-2:], [SystemState.ERROR.value, SystemState.STOPPED.value])

    def test_structured_light_failure_is_structured_failure(self):
        controller = build_system("mock")
        controller.structured_light_runner.fail = True
        result = controller.run_once()
        self.assertFalse(result.success)
        self.assertEqual(result.failed_state, SystemState.STRUCTURED_LIGHT_SCAN.value)
        self.assertEqual(result.error_type, "RuntimeError")


if __name__ == "__main__":
    unittest.main()
