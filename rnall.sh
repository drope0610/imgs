#!/bin/bash

# Liste de toutes les autres catégories à traiter
CATEGORIES=("cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

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
    ONNX_PATH=$(find ./results/efficientad/$CAT -name "model.onnx" | head -n 1)

    if [ -z "$ONNX_PATH" ]; then
        echo "❌ Impossible de trouver le fichier ONNX pour $CAT."
        continue
    fi

    echo "🎯 Fichier ONNX trouvé : $ONNX_PATH"
    echo "🛠️ Compilation du moteur TensorRT (FP16) via trtexec..."

    # 3. Compilation TensorRT optimisée
    /usr/src/tensorrt/bin/trtexec \
        --onnx=$ONNX_PATH \
        --saveEngine=./results/efficientad/${CAT}/efficientad_${CAT}.engine \
        --fp16

    if [ $? -eq 0 ]; then
        echo "✅ [SUCCÈS GLOBAL] Moteur TensorRT créé pour $CAT !"
    else
        echo "❌ Échec de la compilation TensorRT pour $CAT."
    fi
done

echo "🎉 TOUTES LES CATÉGORIES ONT ÉTÉ TRAITÉES AVEC SUCCÈS !"
