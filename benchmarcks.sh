#!/bin/bash

# --- CONFIGURATION DES CHEMINS ---
# On récupère le chemin absolu où se trouve Miniconda
CONDA_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"

echo "=========================================================="
echo "   DEBUT DU PIPELINE D'EVALUATION AUTOMATIQUE             "
echo "=========================================================="

# 1. Chargement des commandes Conda dans le script Bash
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "[ERREUR] Impossible de trouver l'initialisation de Conda à l'emplacement : $CONDA_PATH"
    exit 1
fi

# ==========================================================
# ETAPE 1 : GENERATION DES PREDICTIONS (ANOMALIB / WINCLIP)
# ==========================================================
echo -e "\n>>> 1. Activation de l'environnement Anomalib..."
# Remplace 'base' par le nom de ton environnement si tu en as créé un exprès pour Anomalib
conda activate base  

echo ">>> Execution du script de test WinClip..."
python test_winclip.py

if [ $? -ne 0 ]; then
    echo "[ERREUR] Le script test_winclip.py a plante. Arret du pipeline."
    exit 1
fi

# ==========================================================
# ETAPE 2 : CHANGEMENT D'ENVIRONNEMENT ET EVALUATION MVTEC
# ==========================================================
echo -e "\n>>> 2. Bascule d'environnement Conda pour l'evaluation..."
# Conda va automatiquement désactiver l'environnement précédent pour activer celui-ci
conda activate mad_eval_script

echo ">>> Lancement du script officiel de calcul MVTec AD..."
python mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir mvtec_anomaly__detection \
    --anomaly_maps_dir predictions \
    --output_dir metrics

if [ $? -ne 0 ]; then
    echo "[ERREUR] L'evaluation MVTec AD a echoue."
    exit 1
fi

# ==========================================================
# ETAPE 3 : AFFICHAGE DES SCORES FINAUX
# ==========================================================
echo -e "\n>>> 3. Affichage du bilan des performances (AU-ROC / AU-PRO)..."
python mvtec_ad_evaluation/print_metrics.py --metrics_folder ./metrics/

# Désactivation finale pour rendre le terminal propre à l'utilisateur
conda deactivate

echo -e "\n=========================================================="
echo "   PIPELINE TERMINE AVEC SUCCES !                         "
echo "=========================================================="
