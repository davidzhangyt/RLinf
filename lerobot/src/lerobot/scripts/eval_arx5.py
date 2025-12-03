import os
import ctypes


def ensure_modern_libstdcpp():
    """Load a recent libstdc++ before any heavy deps import it."""
    if os.environ.get("LEROBOT_LIBSTDCXX_PRELOADED") == "1":
        return

    candidate_paths = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate_paths.append(os.path.join(conda_prefix, "lib", "libstdc++.so.6"))
    candidate_paths.append("/home/yueteng/anaconda3/envs/arx-py310/lib/libstdc++.so.6")

    for lib_path in candidate_paths:
        if not lib_path or not os.path.exists(lib_path):
            continue
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            os.environ["LEROBOT_LIBSTDCXX_PRELOADED"] = "1"
            print(f"[INFO] Preloaded modern libstdc++ from {lib_path}")
            return
        except OSError as exc:
            print(f"[WARN] Failed to preload {lib_path}: {exc}")

    print("[WARN] Could not preload modern libstdc++; continuing anyway")


ensure_modern_libstdcpp()

import time
import torch
import numpy as np
import logging
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.arx5_follower.arx5_follower import Arx5Follower
from lerobot.robots.arx5_follower.config_arx5_follower import Arx5FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.utils import init_logging

# 配置部分
# 模型路径
PRETRAINED_PATH = "/home/yueteng/RLinf/lerobot/pi0_train/pretrained_model"
# 机器人接口
ROBOT_INTERFACE = "can0"
# 摄像头配置
CAMERA_FRONT_PORT = "/dev/video6"
CAMERA_ARM_PORT = "/dev/video12"
# 控制频率 (Hz)
CONTROL_FREQ = 30

def main():
    init_logging()
    logging.info("Initializing...")
    
    # 1. 配置机器人
    robot_config = Arx5FollowerConfig(
        interface=ROBOT_INTERFACE,
        cameras={
            "front": OpenCVCameraConfig(index_or_path=CAMERA_FRONT_PORT, width=640, height=480, fps=30),
            "arm": OpenCVCameraConfig(index_or_path=CAMERA_ARM_PORT, width=640, height=480, fps=30),
        },
        model="L5" # 默认为L5，如果是X5请修改
    )
    
    # 2. 连接机器人
    logging.info("Connecting to robot...")
    robot = Arx5Follower(robot_config)
    robot.connect()
    logging.info("Robot connected.")
    
    # 3. 加载策略
    logging.info("Loading policy...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(PRETRAINED_PATH)
    policy.to(device)
    policy.eval()
    policy.config.device = device
    logging.info("Policy loaded.")
    
    # 4. 加载预处理器
    # 注意：这里会自动加载训练时的统计数据(stats)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    
    # 策略输入配置
    # 注意：pi0模型需要文本提示，根据任务修改
    TASK_DESCRIPTION = "Push the T-block to the target"

    logging.info("Starting inference loop. Press Ctrl+C to stop.")
    dt = 1.0 / CONTROL_FREQ
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # A. 获取观测 (Observation)
            # robot_obs keys: 'joint_0.pos', ..., 'gripper.pos', 'front', 'arm'
            robot_obs = robot.get_observation()
            
            # B. 构建符合 Policy 输入格式的 Batch (EnvTransition)
            
            # 1. 状态向量 (Joints + Gripper) -> 'observation.state'
            # 假设训练时 state 是 [joint_0, ..., joint_5, gripper]
            joint_pos = [robot_obs[f"joint_{i}.pos"] for i in range(6)]
            joint_pos.append(robot_obs["gripper.pos"])
            # 注意：不需要手动添加 batch 维度，preprocessor 会处理
            state_tensor = torch.tensor(joint_pos, dtype=torch.float32, device=device) # (7,)
            
            # 2. 图像数据 -> 'observation.images.xxx'
            # 机器人返回的是 (H, W, C) uint8 numpy array
            # 需要转换为 (C, H, W) float32 [0, 1] tensor
            
            def process_image(img_array):
                tensor = torch.from_numpy(img_array).to(device, dtype=torch.float32)
                tensor = tensor.permute(2, 0, 1) # HWC -> CHW
                tensor = tensor / 255.0 # 归一化到 0-1
                return tensor # (C, H, W)
            
            # 映射关系 (根据训练脚本 pushT_train.sh):
            # front -> base_0_rgb
            # arm -> right_wrist_0_rgb
            
            img_front = process_image(robot_obs["front"])
            img_arm = process_image(robot_obs["arm"])
            
            # 构建 dataset 风格的 Batch（扁平键）
            transition = {
                "observation.state": state_tensor,
                "observation.images.base_0_rgb": img_front,
                "observation.images.right_wrist_0_rgb": img_arm,
                "task": TASK_DESCRIPTION,
            }
            
            # C. 预处理 (归一化, Tokenize, Add Batch Dim 等)
            # preprocessor 期望 EnvTransition 结构
            batch = preprocessor(transition)
            
            # D. 推理
            with torch.inference_mode():
                # pi0 策略期望扁平的 observation 字典作为输入
                # 此时 batch 已经是扁平结构，直接传给 select_action
                output_action = policy.select_action(batch)
            
            # E. 后处理 (反归一化)
            # postprocessor 期望 EnvTransition 结构 {"action": ...}
            action_transition = {"action": output_action}
            action_transition = postprocessor(action_transition)
            raw_action = action_transition["action"]
            
            # F. 发送动作
            # raw_action shape: (1, action_dim) -> (action_dim,)
            # 注意：如果模型输出 padding 过的 32 维动作，我们只需要前 7 维
            action_vec = raw_action.squeeze(0).cpu().numpy()
            
            # 构建发送给机器人的动作字典
            action_dict = {}
            for i in range(6):
                action_dict[f"joint_{i}.pos"] = action_vec[i]
            action_dict["gripper.pos"] = action_vec[6]
            
            robot.send_action(action_dict)
            
            # G. 维持频率
            process_time = time.perf_counter() - loop_start
            sleep_time = dt - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logging.warning(f"Loop overrun: {process_time*1000:.1f}ms > {dt*1000:.1f}ms")

    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    finally:
        robot.disconnect()
        logging.info("Disconnected.")

if __name__ == "__main__":
    main()

