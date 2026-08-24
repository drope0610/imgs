#!/bin/bash

# Script to batch train EfficientAD models on pill and pillqc datasets
# Categories: pill, pillqc
# Image sizes: 256, 512
# Epochs: 50, 250, 500

# Stop on first error
set -e

# Export python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# Activate environment if it exists
if [ -f "$HOME/anomalib_env/bin/activate" ]; then
    source "$HOME/anomalib_env/bin/activate"
fi

echo "=========================================================="
echo "🚀 DÉMARRAGE DE L'ENTRAÎNEMENT PAR LOTS (EFFICIENT AD) 🚀"
echo "=========================================================="

CATEGORIES=("pill" "pillqc")
IMG_SIZES=(256 512)
EPOCHS_LIST=(50 250 500)

for CATEGORY in "${CATEGORIES[@]}"; do
    if [ "$CATEGORY" == "pill" ]; then
        DATASET_TYPE="mvtec"
    else
        DATASET_TYPE="folder"
    fi

    for IMG_SIZE in "${IMG_SIZES[@]}"; do
        for EPOCHS in "${EPOCHS_LIST[@]}"; do
            RUN_NAME="efficientad_${CATEGORY}_${IMG_SIZE}_${EPOCHS}ep"
            
            echo ""
            echo "----------------------------------------------------------"
            echo "🔥 Entraînement: Catégorie: $CATEGORY | Taille: ${IMG_SIZE}x${IMG_SIZE} | Epoques: $EPOCHS"
            echo "📂 Nom du run: $RUN_NAME"
            echo "----------------------------------------------------------"
            
            # Lancement de l'entraînement
            python3 src/train.py \
                --category "$CATEGORY" \
                --dataset_type "$DATASET_TYPE" \
                --img_size "$IMG_SIZE" \
                --epochs "$EPOCHS" \
                --run_name "$RUN_NAME"
                
            echo "✅ $RUN_NAME terminé avec succès !"
            echo "----------------------------------------------------------"
        done
    done
done

echo ""
echo "🎉 TOUS LES ENTRAÎNEMENTS SONT TERMINÉS ! 🎉"
