#!/bin/bash
# source ~/anaconda3/bin/activate arx-py310
# Get the directory where the script is located
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=${HF_TOKEN:-""}
export HF_HOME=/home/yueteng/RLinf/hf_cache
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure PYTHONPATH includes current directory and ARX SDK (relative to script)
# Same as pushT.sh - this is what made data collection work!
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR:$SCRIPT_DIR/../../arx5-sdk/python

# Fix libstdc++ version mismatch by FORCING conda's newer version
# Use CONDA_PREFIX if available, otherwise try common conda path
if [ -n "$CONDA_PREFIX" ]; then
    CONDA_LIB="$CONDA_PREFIX/lib"
else
    # Fallback to your specific conda environment path
    CONDA_LIB="/home/yueteng/anaconda3/envs/arx-py310/lib"
fi

# Use LD_PRELOAD to force loading conda's libstdc++ before anything else
export LD_PRELOAD=$CONDA_LIB/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_LIB:$LD_LIBRARY_PATH
echo "[INFO] Using LD_PRELOAD: $LD_PRELOAD"

# Run the evaluation script
python $SCRIPT_DIR/lerobot/scripts/eval_arx5.py "$@"

