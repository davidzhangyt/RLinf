# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("arx5_follower")
@dataclass
class Arx5FollowerConfig(RobotConfig):
    # ARX arm model: L5 or X5
    model: str = "L5"
    # CAN interface name, e.g., can0
    interface: str = "can0"
    
    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Mock mode for testing without robot
    mock: bool = False

