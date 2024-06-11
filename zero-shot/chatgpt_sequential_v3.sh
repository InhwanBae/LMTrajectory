#!/bin/bash
echo "Start chatgpt sequential queues"

# Hyperparameters
dataset=1
model=0
start_id=0
end_id=10000

# Arguments
while getopts d:m: flag
do
  case "${flag}" in
    d) dataset=${OPTARG};;
    m) model=${OPTARG};;
    *) echo "usage: $0 [-d DATASET_ID] [-m LLM_MODEL_ID]" >&2
      exit 1 ;;
  esac
done

# Start training tasks
printf "Evaluate dataset id ${dataset}\n"
for (( i=${start_id}; i<${end_id}; i++ ))
do
  python3 chatgpt_trajectory_predictor_v3.py --dataset "${dataset}" --model "${model}" --scene_id "${i}"
done

echo "Done."