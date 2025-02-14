#! /bin/bash 

set -e 

# Start grobid if session not already exists 
if tmux has-session -t run_grobid 2>/dev/null; then
    echo "Grobid is already running"
    sleep 1
else
    echo "Starting Grobid"
    tmux new-session -d -s run_grobid "bash ./scripts/run_grobid.sh"
    # Wait until GORBID started
    sleep 10
fi

# Download section classifier 
mkdir -p model_store

cd model_store
if [ ! -f "section_classifier_openreview.ckpt" ]; then
    gdown 1HWIeDzumNDZFaOT-AvKl8uqSYppdzctA
fi
cd ..

# Run the script to complete the openreview dataset
python pipeline.py --config ./configs/pipeline_v2.yaml

# Clean up the tmux session 
tmux kill-session -t run_grobid