# Anomalib Jetson Benchmarks

Ce projet vise à tester, évaluer et déployer des modèles de détection d'anomalies (notamment **WinClip** et **EfficientAd**) en utilisant la bibliothèque [Anomalib](https://github.com/openvinotoolkit/anomalib). 
L'objectif principal est d'optimiser ces modèles pour une exécution ultra-rapide sur des architectures embarquées, spécifiquement les cartes **NVIDIA Jetson Orin**.

## 🚀 Fonctionnalités Principales

- **Évaluation de WinClip** : Inférence directe pour générer des cartes de chaleur d'anomalies (Anomaly Maps) sur le dataset MVTec.
- **Entraînement d'EfficientAd** : Pipeline d'apprentissage non supervisé sur les différentes catégories industrielles.
- **Déploiement TensorRT** : Export automatisé des modèles PyTorch vers ONNX, puis compilation en moteurs TensorRT (`.engine`) avec quantification **FP16** et **INT8** pour maximiser le framerate (FPS) sur Jetson Orin.
- **Inférence Bas Niveau** : Scripts d'inférence (via `pycuda`) gérant la mémoire Pagelocked pour des transferts Host-to-Device sans goulot d'étranglement.

## 💻 Matériel Cible : NVIDIA Jetson Orin

Les entraînements et les optimisations TensorRT de ce projet sont conçus pour exploiter au maximum l'architecture matérielle de la carte embarquée **NVIDIA Jetson Orin (32GB)** :
- **Architecture GPU** : NVIDIA Ampere (Compute Capability 8.7)
- **Unités de calcul** : 8 Streaming Multiprocessors (SMs)
- **Mémoire Unifiée** : ~32 Go LPDDR5 (Partagée entre CPU et GPU pour éviter les transferts PCIe)
- **Accélération IA** : Tensor Cores actifs (Cruciaux pour l'inférence TensorRT en précision mixte FP16 et INT8)
- **Stockage** : Disque M.2 NVMe ultra-rapide (Réduit le temps de chargement des datasets par 7 comparé à l'USB)

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
├── benchmarks/mvtec_ad_evaluation/       # Sous-module d'évaluation officiel MVTec
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

## 📊 Résultats des Benchmarks (Jetson Orin)

Voici l'historique des optimisations réalisées sur le modèle **WinClip (Zero-Shot)** pour la classification d'images industrielles (Exemple sur la catégorie *Toothbrush* : 42 images, 240x240).

| Version | Configuration | Latence par image | Débit (FPS) | Remarques |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (PyTorch pur)** | FP32, `empty_cache()`, Segmentation (Sliding Windows), Batch=2 | ~ 1.44 s | 0.7 FPS | Lourd, goulot d'étranglement mémoire, pas de Tensor Cores. |
| **Autocast (Mixed Precision)** | FP16 via `autocast`, sans `empty_cache()`, Batch=2 | ~ 1.61 s | 0.6 FPS | **Plus lent.** Le CPU ARM de la Jetson sature en effectuant les conversions dynamiques (Cast) FP16/FP32 exigées par le Transformer. |
| **Full Optimisé (Industriel)** | FP16 natif (`model.half()`), Classification pure (`scales=()`), Batch=1 | **~ 16 à 26 ms** | **~ 40 à 60 FPS** | Utilisation totale des Tensor Cores (sans overhead CPU). Mode classification seule idéale pour les flux temps réel. |

> **Conclusion** : Pour le déploiement Jetson, forcer le réseau en véritable FP16 (plutôt que d'utiliser l'autocast dynamique) et désactiver les cartes thermiques si non requises permet de passer de 1,4 seconde à **16 millisecondes** par image (un gain de vitesse de **x87**).

### Analyses Approfondies en Production (Multi-Jetsons)

Suite au déploiement du code sur **trois cartes Jetson Orin** en parallèle (avec un chargement dynamique des datasets via des clés USB locales), nous avons mené des analyses poussées sur le comportement de WinClip.

#### 1. Latence : Image Saine (OK) vs Défectueuse (NOK)
Nos tests démontrent que **le temps de calcul est strictement identique**, qu'une pièce soit parfaite ou endommagée (mesuré à **14.2 ms** par image sur Jetson 8 cœurs).
**Explication** : L'architecture d'un réseau de neurones (Deep Learning) est statique. Le GPU (Tensor Cores) exécute toujours le même nombre d'opérations matricielles pour traverser les couches du réseau, garantissant un flux vidéo ultra-stable et constant de **~70 FPS**, peu importe la présence de rayures ou de défauts.

#### 2. Calibration Zero-Shot (Seuil F1-Max)
Par défaut, utiliser un seuil arbitraire (`> 0.5`) sur un modèle Zero-Shot donne une précision illusoire et très faible (~25%). Pour révéler la véritable performance du modèle, le pipeline calcule automatiquement le **seuil F1-Max** en utilisant la catégorie des médicaments (`pill`) comme référence de calibration.

En appliquant ce seuil mathématique (`0.3877`) à l'ensemble du dataset industriel, la précision fait un bond spectaculaire sans aucun entraînement supplémentaire :
- **LEATHER** (Cuir) : **99.2%** de réussite
- **GRID** (Grilles) : **94.9%** de réussite
- **CARPET** (Tapis) : **92.3%** de réussite
- **PILL** (Médicaments) : **83.8%** de réussite

#### 3. Politique Industrielle "Zéro Défaut" (100% Recall)
Dans un contexte industriel strict, l'objectif est d'intercepter **100% des pièces défectueuses** (0 Faux Négatif). 
Pour ce faire, l'algorithme a été calibré pour trouver le seuil mathématique le plus intransigeant garantissant qu'aucune anomalie ne passe, en se basant sur la catégorie des médicaments (`pill`). 

**Résultats de la politique Zéro Défaut avec WinClip (Zero-Shot) :**
- **Défauts interceptés** : **100%** (Les 141 médicaments défectueux ont bien été rejetés).
- **Faux Positifs (Pièces saines jetées à tort)** : **96.2%** (Les pièces parfaitement saines ont presque toutes été jetées !).

**Explication** : Le modèle actuel (WinClip) est un algorithme *Zero-Shot* (il n'a jamais été entraîné sur nos pièces spécifiques). Pour s'assurer de détecter les défauts les plus microscopiques, on doit abaisser son seuil de tolérance de manière si drastique qu'il devient "paranoïaque" et considère la moindre variation de texture d'une pièce parfaite comme une anomalie.

**Conclusion** : Une politique "Zéro Défaut" stricte est inapplicable avec un modèle Zero-Shot en production, car elle engendre beaucoup trop de faux rejets (perte sèche). Le passage à un modèle entraîné sur mesure (comme **EfficientAd**) est absolument indispensable pour maintenir 100% d'interception des défauts tout en préservant les pièces saines.

### Benchmarks Spécialisés : Industrie Pharmaceutique
Ces nouveaux tests évaluent la robustesse de WinClip en Zero-Shot sur des datasets complexes liés à la pharmacie et au contrôle qualité des médicaments, exécutés sur la Jetson Orin :

| Benchmark / Dataset | Catégorie | Images Évaluées | Latence (ms/img) | Débit (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **MVTec Pharma** | `pill` | 167 | 32.12 ms | 31.1 FPS |
| **MVTec Pharma** | `capsule` | 132 | 24.66 ms | 40.6 FPS |
| **VisA** | `capsules` | 702 | 34.42 ms | 29.0 FPS |
| **PillQC** | `pill` | 15 | 64.84 ms | 15.4 FPS |

## 📊 Résultats : Efficace (EfficientAD - Sur-mesure)

L'architecture EfficientAD (modèle "Étudiant-Professeur") s'entraîne spécifiquement sur des images de pièces saines pour comprendre la normalité absolue de notre produit (Médicaments / `pill`). Le but est de drastiquement réduire les fausses alertes sans compromettre la politique du Zéro Défaut.

| Entraînement | Latence PyTorch (Natif) | Zéro Défaut (Rappel Défauts) | Faux Positifs (Rejets abusifs) |
| :--- | :--- | :--- | :--- |
| **10 Époques** (Test Validation) | ~60.8 ms | 100% (Validé) | 96.2% (25/26 pièces jetées) |
| **250 Époques** (Test Intermédiaire) | ~60.5 ms | 100% (Validé) | **69.2%** (18/26 pièces jetées) |
| **1000 Époques** (Test Marathon 26h) | ~57.8 ms | 100% (Validé) | **88.5%** (23/26 pièces jetées) ⚠️ (Surapprentissage) |
| **5 Époques (Optimisation NVMe)** | **~56.3 ms** | 100% (Validé) | 92.3% (24/26 pièces jetées) |

*Note : Les résultats de l'entraînement extrême de 1000 époques révèlent une dégradation des performances due au phénomène de **surapprentissage (overfitting)**. L'architecture mathématique d'EfficientAD imposant un `batch_size` de 1, le modèle a fini par mémoriser par cœur les pièces saines du jeu d'entraînement, devenant hyper-sensible et rejetant de nouveau massivement les pièces saines lors des tests. La configuration à **250 époques** semble donc être l'optimum "Sweet Spot" pour cette catégorie.*

### ⚡ Optimisation Matérielle : Disque NVMe 1To
Pour palier aux goulots d'étranglement de la clé USB, le dataset a été migré vers un disque dur M.2 NVMe ultra-rapide sur la **Jetson Orin (Pedro2)**. Les résultats de débit des données sont fulgurants :
- **Temps de chargement (Entraînement EfficientAD)** : Passage de ~35 secondes par époque (USB) à seulement **5 secondes par époque** (NVMe), soit une vitesse de chargement multipliée par 7 !
- **Latence d'Inférence pure (EfficientAD)** : Très légère amélioration à **~56.3 ms**.
- **Latence d'Inférence (WinCLIP Zero-Shot)** : Débit maintenu à environ **35 FPS** (limitée par la puissance de calcul GPU et non plus par le disque).

---
*Ce document est évolutif et sera enrichi au fur et à mesure des tests et des avancées sur l'architecture Jetson Orin.*
