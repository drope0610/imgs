import os
from pathlib import Path

# Chemins possibles pour le dataset MVTec
POSSIBLE_DATASET_PATHS = [
    Path("/media/pedro2/nvme/mvtec_anomaly_detection"),
    Path("/media/pedro2/Modeles/mvtec_anomaly_detection"),
    Path("./datasets/mvtec_anomaly_detection"),
    Path("./datasets"),
    Path("/Users/pedro/Desktop/Inria/imgs/datasets/mvtec_anomaly_detection") # Fallback local Mac
]

def get_dataset_root() -> Path:
    """Retourne le premier chemin valide vers le dataset MVTec."""
    for p in POSSIBLE_DATASET_PATHS:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError("Impossible de trouver le dataset MVTec sur le NVMe, les clés USB ou en local.")

def get_results_dir() -> Path:
    """Retourne le répertoire racine des résultats (checkpoints, onnx, engine)."""
    # Résout dynamiquement selon l'environnement (Jetson ou Local Mac)
    if os.path.exists("/home/pedro2/results"):
        return Path("/home/pedro2/results")
    return Path.cwd() / "results"
