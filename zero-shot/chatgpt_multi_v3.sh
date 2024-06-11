#!/bin/bash
echo "Start chatgpt multi queues"

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

# Signal handler
pid_array=()

sighdl ()
{
  echo "Kill training processes"
  for (( i=0; i<${end_id}-${start_id}; i++ ))
  do
    kill ${pid_array[$i]}
  done
  echo "Done."
  exit 0
}

trap sighdl SIGINT SIGTERM

# Start training tasks
for (( i=${start_id}; i<${end_id}; i++ ))
do
  printf "Evaluate dataset id ${dataset} scene id ${i}"
  python3 chatgpt_trajectory_predictor_v3.py --dataset "${dataset}" --model "${model}" --scene_id "${i}" &
  pid_array[$i]=$!
  printf " job ${#pid_array[@]} pid ${pid_array[$i]}\n"
  sleep 0.01
done

for (( i=0; i<${end_id}-${start_id}; i++ ))
do
  wait ${pid_array[$i]}
done

echo "Done."