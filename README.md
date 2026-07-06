# Anomalib Jetson Benchmarks

Ce projet vise à tester, évaluer et déployer des modèles de détection d'anomalies (notamment **WinClip** et **EfficientAd**) en utilisant la bibliothèque [Anomalib](https://github.com/openvinotoolkit/anomalib). 
L'objectif principal est d'optimiser ces modèles pour une exécution ultra-rapide sur des architectures embarquées, spécifiquement les cartes **NVIDIA Jetson Orin**.

## 🚀 Fonctionnalités Principales

- **Évaluation de WinClip** : Inférence directe pour générer des cartes de chaleur d'anomalies (Anomaly Maps) sur le dataset MVTec.
- **Entraînement d'EfficientAd** : Pipeline d'apprentissage non supervisé sur les différentes catégories industrielles.
- **Déploiement TensorRT** : Export automatisé des modèles PyTorch vers ONNX, puis compilation en moteurs TensorRT (`.engine`) avec quantification **FP16** et **INT8** pour maximiser le framerate (FPS) sur Jetson Orin.
- **Inférence Bas Niveau** : Scripts d'inférence (via `pycuda`) gérant la mémoire Pagelocked pour des transferts Host-to-Device sans goulot d'étranglement.

## 📂 Structure du Projet

```text
.
├── scripts/                   # Scripts d'automatisation (Bash)
│   ├── run_all_training.sh    # Entraînement global et génération des moteurs TensorRT
│   └── run_benchmarks.sh      # Pipeline complet d'évaluation MVTec
├── src/                       # Code source Python
│   ├── train.py               # Entraînement d'EfficientAd et export ONNX
│   ├── test_winclip.py        # Évaluation et génération de masques WinClip
│   ├── benchmark/             # Outils de profilage (FPS, Latence)
│   │   └── perf_tester.py
│   └── deploy/                # Code d'inférence optimisé pour le matériel cible
│       └── infer_tensorrt.py
├── mvtec_ad_evaluation/       # Sous-module d'évaluation officiel MVTec
└── requirements.txt           # Dépendances Python nécessaires
```

## 🛠️ Prérequis et Installation

Assurez-vous de disposer d'un environnement Conda ou d'un environnement virtuel Python propre.

1. **Installer les dépendances Python :**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note** : Le projet nécessite `numpy<2.0.0` pour éviter les conflits (_ARRAY_API not found_) avec certaines versions compilées de PyTorch/Anomalib.

2. **Dépendances Matérielles (Pour déploiement sur Jetson Orin) :**
   - NVIDIA JetPack SDK installé (incluant CUDA et cuDNN).
   - **TensorRT** : La commande `trtexec` doit être disponible (le script d'entraînement la cherchera automatiquement).
   - Module `pycuda` installé dans l'environnement pour le script d'inférence.

## 🏃‍♂️ Utilisation

### 1. Entraînement global et Génération TensorRT
Le script suivant parcourt toutes les catégories MVTec configurées, lance l'entraînement d'EfficientAd, génère les fichiers ONNX, puis compile les moteurs TensorRT :
```bash
bash scripts/run_all_training.sh
```

### 2. Évaluation du modèle WinClip
Ce pipeline lance l'inférence WinClip sur le dataset, génère les prédictions, puis calcule les métriques de performance officielles (AU-ROC / AU-PRO) :
```bash
bash scripts/run_benchmarks.sh
```

### 3. Inférence TensorRT (Mode Production)
Pour tester le moteur compilé sur une image cible via PyCUDA et générer la carte de chaleur visuelle :
```bash
python src/deploy/infer_tensorrt.py
```

---
*Ce document est évolutif et sera enrichi au fur et à mesure des tests et des avancées sur l'architecture Jetson Orin.*
