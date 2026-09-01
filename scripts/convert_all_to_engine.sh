#!/bin/bash
# Convert all ONNX models to TensorRT engines using FP16 optimization

echo "Starting TensorRT Conversion on $(hostname)..."
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH

# Chercher dans les résultats potentiels
find ~/Desktop/Images/imgs/results ~/results -name '*.onnx' 2>/dev/null | while read onnx_path; do
    dir_path=$(dirname "$onnx_path")
    engine_path="$dir_path/model.engine"
    
    if [[ "$onnx_path" == *"512"* ]]; then
        SHAPE="input:1x3x512x512"
    else
        SHAPE="input:1x3x256x256"
    fi
    
    if [ ! -f "$engine_path" ] || [ $(stat -c%s "$engine_path") -lt 1000 ]; then
        echo "Converting $onnx_path -> $engine_path (Shape: $SHAPE)"
        /usr/src/tensorrt/bin/trtexec --onnx="$onnx_path" --saveEngine="$engine_path" --fp16 --minShapes=$SHAPE --optShapes=$SHAPE --maxShapes=$SHAPE > "$dir_path/trt_conversion.log" 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Successfully converted $engine_path"
        else
            echo "❌ Failed to convert $onnx_path"
        fi
    else
        echo "⏭️ $engine_path already exists, skipping."
    fi
done

echo "Conversion complete on $(hostname)!"
