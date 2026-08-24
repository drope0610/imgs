# 💊 Benchmarks de Détection d'Anomalies Pharmaceutiques

Ce dossier contient une suite complète de benchmarks standardisés pour évaluer vos propres modèles de détection d'anomalies visuelles dans le domaine des médicaments et de l'industrie pharmaceutique.

---

## 📁 Benchmarks Disponibles

| Fichier | Benchmark / Dataset | Domaine d'Application | Catégories Ciblées |
| :--- | :--- | :--- | :--- |
| [`mvtec_pharma.py`](file:///Users/pedro/Desktop/Inria/imgs/benchmarks/mvtec_pharma.py) | **MVTec AD** | Industrie / Production | `pill` (pilules) & `capsule` (gélules) |
| [`visa_capsules.py`](file:///Users/pedro/Desktop/Inria/imgs/benchmarks/visa_capsules.py) | **Amazon VisA** | Anomalies complexes | `capsules` |
| [`pillqc.py`](file:///Users/pedro/Desktop/Inria/imgs/benchmarks/pillqc.py) | **PillQC** | Contrôle Qualité | Pilules saines, contaminations (dirt), éclats (chip) |

---

## 🚀 Utilisation Rapide

Chaque script accepte des arguments standard pour passer votre modèle (`--model_path`), définir le chemin du jeu de données (`--dataset_dir`) et choisir l'appareil de calcul (`--device`).

### 1. MVTec AD Pharmaceutique (Pilules & Gélules)
```bash
python benchmarks/mvtec_pharma.py \
    --dataset_dir datasets/mvtec_anomaly_detection \
    --categories pill capsule \
    --model_path vos_modeles/efficientad_pill.pt \
    --device cuda
```

### 2. VisA Capsules
```bash
python benchmarks/visa_capsules.py \
    --dataset_dir /chemin/vers/visa \
    --category capsules \
    --model_path vos_modeles/mon_modele.pt
```

### 3. PillQC (Quality Control)
```bash
python benchmarks/pillqc.py \
    --dataset_dir /chemin/vers/pillqc \
    --model_path vos_modeles/mon_modele.pt
```


---

## 📊 Format des Fichiers de Sortie (CSV)

Chaque benchmark génère automatiquement un fichier CSV séparé par des points-virgules (compatible Excel) contenant :
- Nom du fichier image
- Vérité terrain (`OK` / `NOK`)
- Score d'anomalie calculé
- Latence d'inférence brute (en millisecondes)
- Débit global (FPS)
