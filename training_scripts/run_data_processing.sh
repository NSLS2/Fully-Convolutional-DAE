#!/bin/bash

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Submits a data processing job"
    echo "Usage: $0 <mode>"
    echo "<mode> - bootstrap | normal"
    exit 0
fi

mode=${1:-normal}    # first argument (default normal)
node=$(hostname | cut -d. -f1)

echo "Mode: $mode"
echo "Node: $node"

if [[ "$mode" == "bootstrap" ]]; then
    base_cmd="python data_processing.py --input ../data/ --enable_bootstrap --bs_data_path ../bs_data_corrected --output ../processed_data"
    suffix="_bs"
elif [[ "$mode" == "normal" ]]; then
    base_cmd="python data_processing.py --input ../data/ --output ../processed_data"
    suffix=""
else
    echo "You should choose a mode: bootstrap | normal"
    exit 1
fi

# Loop over seeds
for step in "test" "train"; do
    cmd="${base_cmd}${suffix}_${node} --step ${step}"
    echo "Running: $cmd"
    $cmd
done
