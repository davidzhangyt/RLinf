import importlib
import logging
import os
import sys
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import ctypes
import numpy as np


def _preload_modern_libstdcpp():
    """Fallback safety: ensure libstdc++ is new enough if not already done."""
    if os.environ.get("LEROBOT_LIBSTDCXX_PRELOADED") == "1":
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "lib", "libstdc++.so.6"))
    candidates.append("/home/yueteng/anaconda3/envs/arx-py310/lib/libstdc++.so.6")

    for lib_path in candidates:
        if not lib_path or not os.path.exists(lib_path):
            continue
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            os.environ["LEROBOT_LIBSTDCXX_PRELOADED"] = "1"
            logging.info("Preloaded modern libstdc++ from %s", lib_path)
            return
        except OSError:
            continue


_preload_modern_libstdcpp()
# Try to import arx5_interface, if not found, user needs to set PYTHONPATH
def _import_arx5_interface() -> Any:
    """
    Attempt to import arx5_interface by searching common SDK locations.
    Returns the module on success, None otherwise.
    """

    # 1) Respect explicit override
    env_path = os.environ.get("ARX5_SDK_PATH")
    search_paths = [Path(env_path)] if env_path else []

    # 2) Scan up the directory tree to find arx5-sdk
    # This handles various nesting levels (e.g. in monorepo or standalone)
    current_path = Path(__file__).resolve()
    print(f"[DEBUG] Current file: {current_path}")
    for parent in current_path.parents:
        sdk_candidate = parent / "arx5-sdk" / "python"
        print(f"[DEBUG] Checking: {sdk_candidate}, exists={sdk_candidate.exists()}")
        if sdk_candidate.exists():
            search_paths.append(sdk_candidate)
            print(f"[DEBUG] Found SDK at: {sdk_candidate}")
            break
    
    # 3) Current working directory (legacy/fallback)
    search_paths.append(Path.cwd() / "arx5-sdk" / "python")
    
    # 4) Dynamic relative path finding (Project structure specific)
    # We assume the structure: root/lerobot/.../arx5_follower.py and root/arx5-sdk/python
    # We need to go up 6 levels from this file's directory to reach 'root'
    # lerobot/src/lerobot/robots/arx5_follower/ -> arx5_follower -> robots -> lerobot -> src -> lerobot -> root
    try:
        # Use .resolve() to get absolute path, then go up parents
        current_file = Path(__file__).resolve()
        # parents[0] is dir, parents[1] is robots, ... parents[5] is root (RLinf)
        if len(current_file.parents) > 5:
            workspace_root = current_file.parents[5]
            search_paths.append(workspace_root / "arx5-sdk" / "python")
    except Exception:
        pass

    print(f"[DEBUG] All search paths: {search_paths}")
    for path in search_paths:
        if not path or not path.exists():
            continue
        abs_path = str(path.resolve())
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)  # 插入到最前面，优先级更高
        try:
            module = importlib.import_module("arx5_interface")
            print(f"[DEBUG] Successfully imported arx5_interface from {abs_path}")
            return module
        except ImportError as e:
            print(f"[DEBUG] Import failed from {abs_path}: {e}")
            continue
        except Exception as e:
            print(f"[DEBUG] Unexpected error importing from {abs_path}: {type(e).__name__}: {e}")
            continue

    return None


try:
    arx5 = importlib.import_module("arx5_interface")
except ImportError:
    arx5 = _import_arx5_interface()

from ..robot import Robot
from .config_arx5_follower import Arx5FollowerConfig
from lerobot.cameras.utils import make_cameras_from_configs

logger = logging.getLogger(__name__)

class Arx5Follower(Robot):
    config_class = Arx5FollowerConfig
    name = "arx5_follower"

    def __init__(self, config: Arx5FollowerConfig):
        super().__init__(config)
        self.config = config
        self._controller = None
        self._robot_config = None
        self._connected = False
        
        if arx5 is None and not config.mock:
            raise ImportError("Could not import arx5_interface. Please ensure arx5-sdk is installed or in PYTHONPATH.")

        self.cameras = make_cameras_from_configs(config.cameras)
        self.joint_names = [] # Will be populated after connection or based on model
        
        # Gripper mapping parameters (from collect_data_with_camera.py)
        self.master_gripper_min = 0.0001
        self.master_gripper_max = 0.0211
        self.slave_gripper_min = 0.0
        self.slave_gripper_max = 0.08 # Will be updated from robot config
    
    def map_gripper(self, master_gripper_raw: float) -> float:
        """Map master gripper position to slave range with fixed range (no dynamic update)."""
        # Fixed range based on calibration
        # if master_gripper_raw < self.master_gripper_min:
        #     self.master_gripper_min = master_gripper_raw
        # if master_gripper_raw > self.master_gripper_max:
        #     self.master_gripper_max = master_gripper_raw
        
        # Linear mapping
        if self.master_gripper_max > self.master_gripper_min:
            ratio = (master_gripper_raw - self.master_gripper_min) / \
                   (self.master_gripper_max - self.master_gripper_min)
        else:
            ratio = 0.5
        
        slave_gripper = self.slave_gripper_min + ratio * \
                       (self.slave_gripper_max - self.slave_gripper_min)
        return np.clip(slave_gripper, 0.0, self.slave_gripper_max)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        if self.config.mock:
            logger.info("Mock connection for Arx5Follower")
            self._connected = True
            return

        logger.info(f"Connecting to ARX5 {self.config.model} on {self.config.interface}")
        
        # Initialize controller
        # We use the Factory pattern from SDK
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.config.model)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        
        # Enable background thread for smoother communication
        controller_config.background_send_recv = True
        controller_config.gravity_compensation = True
        self._controller = arx5.Arx5JointController(
            robot_config, controller_config, self.config.interface
        )
        
        # Set log level
        self._controller.set_log_level(arx5.LogLevel.ERROR)
        
        self._robot_config = self._controller.get_robot_config()
        self._connected = True
        
        # Update slave gripper max from config
        self.slave_gripper_max = self._robot_config.gripper_width
        
        # Set default gain for position control
        # Without this, the robot motors won't apply torque to reach target positions
        gain = arx5.Gain(self._robot_config.joint_dof)
        self._controller.set_gain(gain)
        
        # Give the background thread some time to initialize and send the first packets
        time.sleep(1.0)
        
        # Define joint names based on DOF
        dof = self._robot_config.joint_dof
        self.joint_names = [f"joint_{i}" for i in range(dof)]
        # Gripper is handled separately in SDK but we can treat it as an action key
        self.joint_names.append("gripper")
        
        # Initialize cameras
        for cam in self.cameras.values():
            cam.connect()
            
        if calibrate:
            self.calibrate()
            
        logger.info("Arx5Follower connected.")

    def calibrate(self) -> None:
        if not self.is_connected or self.config.mock:
            return
            
        logger.info("Calibrating ARX5 Follower...")
        # ARX SDK has its own calibration/homing
        # Usually resetting to home is a good start
        # We restore this to ensure the robot is enabled and in a known state
        self._controller.reset_to_home()
        
        # Calibrate gripper
        logger.info("Skipping gripper calibration as per user request (gripper connectivity issue).")
        # self._controller.calibrate_gripper()
        
        # Mark as calibrated (Lerobot base class doesn't strictly track this boolean unless we do)
        pass

    @property
    def is_calibrated(self) -> bool:
        return True # ARX SDK handles internal calibration state

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
             if self.config.mock:
                 return {}
             raise RuntimeError("Not connected")

        # Get joint state
        state = self._controller.get_joint_state()
        pos = state.pos()
        vel = state.vel()
        torque = state.torque()
        gripper_pos = state.gripper_pos
        
        obs = {}
        dof = self._robot_config.joint_dof
        
        for i in range(dof):
            name = f"joint_{i}"
            obs[f"{name}.pos"] = pos[i]
            # Add velocity if needed, but standard Lerobot often just uses pos for simple policies
            # If needed: obs[f"{name}.vel"] = vel[i]
            
        obs["gripper.pos"] = gripper_pos
        
        # Cameras
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
            
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
             if self.config.mock:
                 return action
             raise RuntimeError("Not connected")
             
        # Construct JointState command
        dof = self._robot_config.joint_dof
        cmd = arx5.JointState(dof)
        
        cmd_pos = cmd.pos() # Get reference to array
        
        # Map action dict to array
        # Assuming action keys are "joint_0.pos", etc.
        for i in range(dof):
            key = f"joint_{i}.pos"
            if key in action:
                cmd_pos[i] = action[key]
            else:
                # If missing, maybe hold current position? 
                # Ideally action should be complete.
                pass
                
        if "gripper.pos" in action:
            # Apply mapping logic
            cmd.gripper_pos = self.map_gripper(action["gripper.pos"])
            
        # DEBUG: Print first joint command to verify data flow
        # if dof > 0:
        #     logger.info(f"Sending cmd: {cmd_pos[0]:.3f} ...")

        self._controller.set_joint_cmd(cmd)
        
        # If using background thread, we might need to sleep or just let it run
        # The SDK example sleeps for dt if using background thread
        # But here we are driven by the external loop freq.
        # We just set the command, the background thread picks it up.
        
        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            return
            
        if not self.config.mock:
            self._controller.set_to_damping()
            # No explicit disconnect method in SDK? 
            # The destructor usually handles it or we just stop sending.
            
        for cam in self.cameras.values():
            cam.disconnect()
            
        self._connected = False

    @cached_property
    def observation_features(self) -> dict:
        # Return structure
        features = {f"joint_{i}.pos": float for i in range(6)} # Assuming 6DOF + gripper
        features["gripper.pos"] = float
        
        # Add camera features as (h, w, c) tuples
        for cam_key, cam_config in self.config.cameras.items():
            features[cam_key] = (cam_config.height, cam_config.width, 3)
            
        return features
        
    @cached_property
    def action_features(self) -> dict:
        return {f"joint_{i}.pos": float for i in range(6)}

    def configure(self) -> None:
        pass

