import os
from pathlib import Path

# Chemins possibles pour le dataset MVTec
POSSIBLE_DATASET_PATHS = [
    Path("/media/pedro2/nvme/mvtec_anomaly_detection"),
    Path("/media/pedro2/Modeles/mvtec_anomaly_detection"),
    Path.home() / "Desktop" / "Images" / "imgs" / "datasets" / "mvtec_anomaly_detection",
    Path.home() / "Desktop" / "Images" / "imgs" / "datasets",
    Path("./datasets/mvtec_anomaly_detection"),
    Path("./datasets"),
    Path("/Users/pedro/Desktop/Inria/imgs/datasets/mvtec_anomaly_detection") # Fallback local Mac
]

def get_dataset_root() -> Path:
    """Retourne le premier chemin valide vers le dataset."""
    for p in POSSIBLE_DATASET_PATHS:
        if p.exists() and p.is_dir():
            return p
    # Default to local path if none exist (might trigger download if valid)
    return Path.home() / "Desktop" / "Images" / "imgs" / "datasets"

def get_results_dir() -> Path:
    """Retourne le répertoire racine des résultats (checkpoints, onnx, engine)."""
    return Path.cwd() / "results"
