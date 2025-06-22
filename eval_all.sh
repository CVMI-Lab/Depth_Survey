#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <method_name>"
    echo "Example: $0 depthanythingv2"
    exit 1
fi

METHOD_NAME=$1
CONFIG_DIR="./configs/moge_benchmark/${METHOD_NAME}"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist!"
    exit 1
fi

DATASETS=("DDAD" "DIODE" "ETH3D" "GSO" "HAMMER" "iBims-1" "KITTI" "NYUv2" "Sintel" "Spring")

for DATASET in "${DATASETS[@]}"; do
    CONFIG_FILE="${METHOD_NAME}_${DATASET}.yaml"
    FULL_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
    
    if [ -f "$FULL_PATH" ]; then
        echo "Testing on ${DATASET}..."
        python eval.py "$FULL_PATH"
        
        if [ $? -ne 0 ]; then
            echo "Error occurred while testing on ${DATASET}"
            # exit 1
        fi
    else
        echo "Config file ${CONFIG_FILE} not found, skipping ${DATASET}..."
    fi
done

echo "All tests completed."