#!/bin/bash
set -e

echo "🚀 Création de l'environnement virtuel pour Anomalib sur Jetson Orin..."

# 1. Créer un environnement virtuel propre
python3 -m venv ~/anomalib_env
source ~/anomalib_env/bin/activate

# 2. Mettre à jour pip, setuptools et wheel
pip install --upgrade pip setuptools wheel

# 3. INSTALLER NUMPY EN PREMIER (Crucial pour bloquer la version)
pip install "numpy<2.0.0"

# 4. Installer PyTorch et Torchvision (Utilisez les wheels NVIDIA pour JetPack si possible)
pip install torch torchvision

# 5. Installer Anomalib sans écraser NumPy
pip install anomalib==1.1.1

echo "✅ Environnement prêt ! Activez-le avec : source ~/anomalib_env/bin/activate"
