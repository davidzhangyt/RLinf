#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure PYTHONPATH includes current directory and ARX SDK (relative to script)
# Assuming arx5-sdk is at ../../arx5-sdk relative to this script in lerobot/src/
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR:$SCRIPT_DIR/../../arx5-sdk/python

# Run the recording script
python -m lerobot.scripts.lerobot_record \
    --robot.type=arx5_follower \
    --robot.interface=can0 \
    --robot.cameras='{
        "front": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30},
        "arm": {"type": "opencv", "index_or_path": "/dev/video10", "width": 640, "height": 480, "fps": 30}
    }' \
    --robot.id=yueteng \
    --robot.calibration_dir=./cali/ \
    --teleop.type=arx5_leader \
    --teleop.interface=can1 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=yueteng/bottle1125 \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the water bottle." \
    --dataset.push_to_hub=False \
    --dataset.private=False \
    --resume=false
