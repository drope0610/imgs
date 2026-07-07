#!/bin/bash

# Configuration Conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

CATEGORIES=("cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
DATASET_DIR="mvtec_anomaly_detection"
PREDICTIONS_DIR="mvtec_anomaly_detection/efficientad_predictions"

echo "=========================================================="
echo "   DEBUT DU BENCHMARK EFFICIENTAD (TENSORRT)              "
echo "=========================================================="

for CAT in "${CATEGORIES[@]}"
do
    ENGINE_PATH="results/engines/efficientad_${CAT}.engine"
    TEST_DIR="${DATASET_DIR}/${CAT}/test"
    OUTPUT_DIR="${PREDICTIONS_DIR}/${CAT}/test"

    if [ -f "$ENGINE_PATH" ]; then
        echo ">>> [TRAITEMENT] Catégorie : ${CAT} (Inférence TensorRT)..."
        python3 src/deploy/evaluate_tensorrt.py \
            --engine "$ENGINE_PATH" \
            --test_dir "$TEST_DIR" \
            --output_dir "$OUTPUT_DIR"
    else
        echo "⚠️ Moteur TensorRT introuvable pour ${CAT}. Pensez à lancer l'entraînement d'abord."
    fi
done

echo -e "\n>>> Évaluation des performances MVTec AD..."
# Nettoyage des anciennes métriques (pour ne pas écraser ou mélanger avec WinClip)
rm -rf ./metrics/

python mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir "$DATASET_DIR" \
    --anomaly_maps_dir "$PREDICTIONS_DIR" \
    --output_dir metrics

echo -e "\n>>> Affichage du bilan (AU-ROC / AU-PRO)..."
python mvtec_ad_evaluation/print_metrics.py --metrics_folder ./metrics/

echo -e "\n>>> Exportation des résultats pour Excel..."
python scripts/export_metrics_csv.py

conda deactivate
echo -e "\n=========================================================="
echo "   BENCHMARK EFFICIENTAD TERMINE !                        "
echo "=========================================================="
