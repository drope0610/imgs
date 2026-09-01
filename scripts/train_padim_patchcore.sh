#!/bin/bash
set -e

# Export python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# Activate environment if it exists
if [ -f "$HOME/anomalib_env/bin/activate" ]; then
    source "$HOME/anomalib_env/bin/activate"
fi

echo "=========================================================="
echo "🚀 DÉMARRAGE DE L'ENTRAÎNEMENT (PADIM & PATCHCORE) 🚀"
echo "=========================================================="

# Configurations
MODELS=("padim" "patchcore")
CATEGORIES=("pill" "pillqc")
IMG_SIZE=256
EPOCHS=1 # Un seul passage est nécessaire pour ces modèles (Feature extraction)

for MODEL in "${MODELS[@]}"; do
    for CATEGORY in "${CATEGORIES[@]}"; do
        
        # Determine dataset type
        DATASET_TYPE="mvtec"
        if [ "$CATEGORY" = "pillqc" ]; then
            DATASET_TYPE="folder"
        fi

        echo "----------------------------------------------------------"
        echo "🔥 Entraînement: Modèle: $MODEL | Catégorie: $CATEGORY | Taille: ${IMG_SIZE}x${IMG_SIZE}"
        echo "----------------------------------------------------------"
        
        python3 src/train_models.py \
            --model "$MODEL" \
            --category "$CATEGORY" \
            --dataset_type "$DATASET_TYPE" \
            --img_size "$IMG_SIZE" \
            --epochs "$EPOCHS"
        
        echo "✅ Modèle $MODEL sur $CATEGORY terminé !"
        
    done
done

echo "🎉 TOUS LES ENTRAÎNEMENTS PADIM ET PATCHCORE SONT TERMINÉS !"
