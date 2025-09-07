#!/bin/bash


batch_sizes=(128 256 1024)
input_script="inference_scripts/resnet_infer.py"
output_dir="results/ncu-reps/resnet"

echo "$input_script"

script_name=$(basename "$input_script" .py)
for batch_size in "${batch_sizes[@]}"; do
    echo "Profiling for batch size $batch_size"
    sudo ncu --target-processes all --set roofline -f -o "$output_dir/${script_name}_bs_${batch_size}_epoch_20_2" bash exp_script.sh "$input_script" "$batch_size"
done
echo "PROFILING SUCCESSFUL!"