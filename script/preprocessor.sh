#!/bin/bash
echo "Start preprocessing task queues"

# Hyperparameters
dataset_array=("eth" "hotel" "univ" "zara1" "zara2")
phase_array=("train" "val" "test")

# Arguments
while getopts d:p: flag
do
  case "${flag}" in
    d) dataset_array=(${OPTARG});;
    p) phase_array=(${OPTARG});;
    *) echo "usage: $0 [-d \"eth hotel univ zara1 zara2\"] [-p \"train val test\"]" >&2
      exit 1 ;;
  esac
done


# Signal handler
pid_array=()

sighdl ()
{
  echo "Kill preprocessing processes"
  for (( i=0; i<${#dataset_array[@]}*${#phase_array[@]}; i++ ))
  do
    kill ${pid_array[$i]}
  done
  echo "Done."
  exit 0
}

trap sighdl SIGINT SIGTERM


# Start training tasks
for (( i=0; i<${#dataset_array[@]}; i++ ))
do
  for (( j=0; j<${#phase_array[@]}; j++ ))
  do
    printf "Preprocess ${dataset_array[$i]} ${phase_array[$j]}"
    python3 utils/preprocessor.py \
    --dataset "${dataset_array[$i]}" \
    --phase "${phase_array[$j]}" &
    pid_array[$i*${#phase_array[@]}+$j]=$!
    printf " job ${#pid_array[@]} pid ${pid_array[$i*${#phase_array[@]}+$j]}\n"
  done
done

for (( i=0; i<${#dataset_array[@]}*${#phase_array[@]}; i++ ))
do
  wait ${pid_array[$i]}
done

echo "Done."