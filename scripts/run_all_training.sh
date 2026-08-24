#!/bin/bash

# Liste de toutes les autres catégories à traiter
CATEGORIES=("capsule" "pill")

# Nombre d'époques souhaité par catégorie
EPOCHS=10

for CAT in "${CATEGORIES[@]}"
do
    echo "================================================================="
    echo "🚀 LANCEMENT DU PIPELINE GLOBAL POUR LA CATÉGORIE : $CAT"
    echo "================================================================="

    # 1. Lancement de l'entraînement et de l'export ONNX
    python3 src/train.py --category $CAT --epochs $EPOCHS
    
    if [ $? -ne 0 ]; then
        echo "❌ Erreur lors de l'entraînement de $CAT. Passage à la catégorie suivante."
        continue
    fi

    # 2. Localisation automatique du fichier ONNX qui vient d'être créé
    RESULTS_DIR=$(python3 -c "from src.config import get_results_dir; print(get_results_dir())")
    ONNX_PATH=$(find $RESULTS_DIR/efficientad/$CAT -name "model.onnx" | head -n 1)

    if [ -z "$ONNX_PATH" ]; then
        echo "❌ Impossible de trouver le fichier ONNX pour $CAT."
        continue
    fi

    echo "🎯 Fichier ONNX trouvé : $ONNX_PATH"
    # 3. Compilation TensorRT optimisée (FP16 & INT8 pour Jetson)
    mkdir -p $RESULTS_DIR/engines
    
    # FP16 Compilation
    echo "⚙️ Compilation du moteur FP16..."
    python3 scripts/compile_trt.py \
        --model=$ONNX_PATH \
        --output=$RESULTS_DIR/engines/efficientad_${CAT}_fp16.engine \
        --category=$CAT

    if [ $? -eq 0 ]; then
        echo "✅ [SUCCÈS] Moteur FP16 créé !"
    else
        echo "❌ Échec de la compilation FP16 pour $CAT."
    fi
    
    # INT8 Compilation
    echo "⚙️ Compilation du moteur INT8..."
    python3 scripts/compile_trt.py \
        --model=$ONNX_PATH \
        --output=$RESULTS_DIR/engines/efficientad_${CAT}_int8.engine \
        --category=$CAT \
        --int8

    if [ $? -eq 0 ]; then
        echo "✅ [SUCCÈS] Moteur INT8 créé !"
        echo "🧹 Nettoyage des modèles PyTorch et ONNX pour libérer de l'espace..."
        rm -rf $RESULTS_DIR/efficientad/${CAT}
    else
        echo "❌ Échec de la compilation INT8 pour $CAT."
    fi
done

echo "🎉 TOUTES LES CATÉGORIES ONT ÉTÉ TRAITÉES AVEC SUCCÈS !"
