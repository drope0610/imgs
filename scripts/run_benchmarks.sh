#!/bin/bash

echo "=========================================================="
echo "   DEBUT DU PIPELINE D'EVALUATION AUTOMATIQUE             "
echo "=========================================================="

# 1. Chargement de l'environnement virtuel
ENV_PATH="$HOME/anomalib_env"
if [ -f "$ENV_PATH/bin/activate" ]; then
    source "$ENV_PATH/bin/activate"
    echo "[INFO] Environnement virtuel activé : $ENV_PATH"
else
    echo "[ERREUR] Impossible de trouver l'environnement virtuel $ENV_PATH."
    exit 1
fi

# Gestion du paramètre de base dir si fourni
DATASET_DIR=${1:-"datasets/mvtec_anomaly_detection"}

# ==========================================================
# ETAPE 1 : GENERATION DES PREDICTIONS (ANOMALIB / WINCLIP)
# ==========================================================
echo -e "\n>>> 1. Execution du script de test WinClip..."
python benchmarks/test_winclip.py

if [ $? -ne 0 ]; then
    echo "[ERREUR] Le script test_winclip.py a plante. Arret du pipeline."
    exit 1
fi

# ==========================================================
# ETAPE 2 : EVALUATION MVTEC
# ==========================================================
echo -e "\n>>> 2. Lancement du script officiel de calcul MVTec AD..."
# Installation des dépendances éventuelles de l'outil d'évaluation
if [ -f "benchmarks/mvtec_ad_evaluation/requirements.txt" ]; then
    pip install -q -r benchmarks/mvtec_ad_evaluation/requirements.txt
fi

python benchmarks/mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir "$DATASET_DIR" \
    --anomaly_maps_dir "$DATASET_DIR/predictions" \
    --output_dir metrics

if [ $? -ne 0 ]; then
    echo "[ERREUR] L'evaluation MVTec AD a echoue."
    exit 1
fi

# ==========================================================
# ETAPE 3 : AFFICHAGE DES SCORES FINAUX
# ==========================================================
echo -e "\n>>> 3. Affichage du bilan des performances (AU-ROC / AU-PRO)..."
python benchmarks/mvtec_ad_evaluation/print_metrics.py --metrics_folder ./metrics/

# Désactivation
deactivate

echo -e "\n=========================================================="
echo "   PIPELINE TERMINE AVEC SUCCES !                         "
echo "=========================================================="

