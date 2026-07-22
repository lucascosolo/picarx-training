"""The sim's synthetic IMU block (SensorSynthesizer._imu_payload) must match the
real world_state.py imu shape and, crucially, turn a NEW collision into a single
'impact' edge - so field_agent's IMU consumer reacts to glass/low-obstacle bumps
in training exactly as it will on the robot."""
import os
import sys
import unittest

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)

from sim.sensors import SensorSynthesizer  # noqa: E402


class _Robot:
    def __init__(self, speed=0.0, yaw=0.0, pan=0.0, tilt=0.0):
        self.actual_speed_cm_s = speed
        self.actual_yaw_rate = yaw       # rad/s
        self.cam_pan_deg = pan
        self.cam_tilt_deg = tilt


class _World:
    def __init__(self, robot, collision_events=0):
        self.robot = robot
        self.collision_events = collision_events


class SimImuPayloadTest(unittest.TestCase):
    def _synth(self, world):
        s = SensorSynthesizer.__new__(SensorSynthesizer)
        s.world = world
        s._prev_collision_events = 0
        return s

    def test_still_robot_reads_idle_and_calibrated(self):
        imu = self._synth(_World(_Robot()))._imu_payload(1000.0)
        self.assertFalse(imu["moving"])
        self.assertFalse(imu["impact"])
        self.assertFalse(imu["tilted"])
        self.assertTrue(imu["calibrated"])
        self.assertFalse(imu["stale"])

    def test_driving_or_turning_is_moving(self):
        self.assertTrue(self._synth(_World(_Robot(speed=20.0)))._imu_payload(1)["moving"])
        self.assertTrue(self._synth(_World(_Robot(yaw=0.5)))._imu_payload(1)["moving"])

    def test_new_collision_is_a_single_impact_edge(self):
        w = _World(_Robot(speed=10.0), collision_events=0)
        s = self._synth(w)
        self.assertFalse(s._imu_payload(1000.0)["impact"])   # no collision yet
        w.collision_events = 1                                # robot hit something
        self.assertTrue(s._imu_payload(1000.5)["impact"])     # rising edge -> impact
        self.assertFalse(s._imu_payload(1001.0)["impact"])    # still 1 -> no re-fire

    def test_head_pose_rides_along(self):
        imu = self._synth(_World(_Robot(pan=30.0, tilt=-10.0)))._imu_payload(1)
        self.assertEqual(imu["head_pose"], {"pan": 30.0, "tilt": -10.0})


if __name__ == "__main__":
    unittest.main()
