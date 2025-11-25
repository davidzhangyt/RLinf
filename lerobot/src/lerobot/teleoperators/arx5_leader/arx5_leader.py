import logging
import time
import numpy as np
import sys
import os
from typing import Any
from functools import cached_property

# Try to import arx5_interface, if not found, user needs to set PYTHONPATH
try:
    import arx5_interface as arx5
except ImportError:
    # Fallback: try to add from known location if present
    sdk_path = os.path.abspath(os.path.join(os.getcwd(), "arx5-sdk", "python"))
    if os.path.exists(sdk_path):
        sys.path.append(sdk_path)
        try:
            import arx5_interface as arx5
        except ImportError:
            arx5 = None
    else:
        arx5 = None

from ..teleoperator import Teleoperator
from .config_arx5_leader import Arx5LeaderConfig

logger = logging.getLogger(__name__)

class Arx5Leader(Teleoperator):
    config_class = Arx5LeaderConfig
    name = "arx5_leader"

    def __init__(self, config: Arx5LeaderConfig):
        super().__init__(config)
        self.config = config
        self._controller = None
        self._robot_config = None
        self._connected = False
        
        if arx5 is None and not config.mock:
            raise ImportError("Could not import arx5_interface. Please ensure arx5-sdk is installed or in PYTHONPATH.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        if self.config.mock:
            logger.info("Mock connection for Arx5Leader")
            self._connected = True
            return

        logger.info(f"Connecting to ARX5 Leader {self.config.model} on {self.config.interface}")
        
        # Initialize controller for leader (similar to follower but used for reading)
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.config.model)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        
        controller_config.background_send_recv = True
        controller_config.gravity_compensation = True
        
        self._controller = arx5.Arx5JointController(
            robot_config, controller_config, self.config.interface
        )
        
        self._controller.set_log_level(arx5.LogLevel.ERROR)
        self._robot_config = self._controller.get_robot_config()
        self._connected = True
        
        # Set to gravity compensation or damping mode for teleoperation
        # Ideally gravity compensation if available and tuned, otherwise damping
        # Users might need to hold the leader arm
        logger.info("Setting Arx5Leader to Gravity Compensation mode (implicitly via config) for lighter interaction.")
        # self._controller.set_to_gravity_compensation() # Not available in API
        # self._controller.set_to_damping() # Commented out to avoid heavy damping
        
        # Explicitly zero out PID gains to ensure no position hold force
        # This allows the gravity compensation to work without fighting position control
        # MOVED: Gain setting must happen AFTER calibration/reset_to_home, because reset_to_home
        # sets its own stiff gains to move the robot.
        
        if calibrate:
            self.calibrate()

        zero_gain = arx5.Gain(
            np.zeros(self._robot_config.joint_dof), # kp
            np.zeros(self._robot_config.joint_dof), # kd
            0.0, # gripper_kp
            0.0  # gripper_kd
        )
        self._controller.set_gain(zero_gain)
        logger.info("Arx5Leader gains set to zero for compliant control (post-calibration).")
            
        logger.info("Arx5Leader connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        if not self.is_connected or self.config.mock:
            return
            
        logger.info("Calibrating ARX5 Leader...")
        # ARX SDK has its own calibration/homing
        self._controller.reset_to_home()
        
        logger.info("Skipping Leader gripper calibration as per user request.")
        # self._controller.calibrate_gripper()

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
             if self.config.mock:
                 return {}
             raise RuntimeError("Not connected")

        # For a leader arm, "action" usually means reading its current state
        # which will be sent as a command to the follower.
        state = self._controller.get_joint_state()
        pos = state.pos()
        gripper_pos = state.gripper_pos
        
        action = {}
        dof = self._robot_config.joint_dof
        
        for i in range(dof):
            action[f"joint_{i}.pos"] = pos[i]
            
        action["gripper.pos"] = gripper_pos
        
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Leader arm usually doesn't receive force feedback in simple setups, 
        # but if haptic feedback is implemented, it would go here.
        pass
        
    def disconnect(self) -> None:
        if not self.is_connected:
            return
            
        if not self.config.mock:
            self._controller.set_to_damping()
            
        self._connected = False

    def configure(self) -> None:
        pass

    @cached_property
    def action_features(self) -> dict:
        return {f"joint_{i}.pos": float for i in range(6)}

    @cached_property
    def feedback_features(self) -> dict:
        return {}
