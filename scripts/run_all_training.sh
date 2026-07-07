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
    echo "🛠️ Détection de trtexec et compilation du moteur TensorRT..."

    # Détection de trtexec
    TRTEXEC_CMD="trtexec"
    if ! command -v $TRTEXEC_CMD &> /dev/null; then
        if [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
            TRTEXEC_CMD="/usr/src/tensorrt/bin/trtexec"
        else
            echo "❌ ERREUR: trtexec introuvable. Veuillez l'ajouter à votre PATH."
            exit 1
        fi
    fi

    # 3. Calibration INT8
    CACHE_FILE="./results/efficientad/${CAT}/efficientad_${CAT}_calib.cache"
    IMAGE_DIR="./mvtec_anomaly_detection/${CAT}/train/good/"
    echo "🧠 Génération du cache de calibration INT8..."
    python3 src/deploy/generate_calibration.py --onnx_path=$ONNX_PATH --image_dir=$IMAGE_DIR --cache_file=$CACHE_FILE

    # 4. Compilation TensorRT optimisée (FP16 & INT8 pour Jetson)
    mkdir -p ./results/engines
    echo "🛠️ Compilation du moteur TensorRT (FP16 & INT8)..."
    $TRTEXEC_CMD \
        --onnx=$ONNX_PATH \
        --saveEngine=./results/engines/efficientad_${CAT}.engine \
        --fp16 --int8 --calib=$CACHE_FILE | tee ./results/perf_tensorrt_${CAT}.txt

    if [ $? -eq 0 ]; then
        echo "✅ [SUCCÈS GLOBAL] Moteur TensorRT créé et benchmarké pour $CAT !"
        echo "🧹 Nettoyage des modèles PyTorch et ONNX pour libérer de l'espace..."
        rm -rf ./results/efficientad/${CAT}
    else
        echo "❌ Échec de la compilation TensorRT pour $CAT."
    fi
done

echo "🎉 TOUTES LES CATÉGORIES ONT ÉTÉ TRAITÉES AVEC SUCCÈS !"
