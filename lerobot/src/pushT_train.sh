#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure PYTHONPATH includes current directory and ARX SDK (relative to script)
# Assuming arx5-sdk is at ../../arx5-sdk relative to this script in lerobot/src/
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR:$SCRIPT_DIR/../../arx5-sdk/python

# Run the recording script
python lerobot/scripts/train.py \
    --dataset.repo_id=yueteng/pusht1128 \
    --policy.type=act \
    --output_dir=outputs/train/arx5_pusht_test \
    --training.batch_size=8 \
    --training.steps=10000
