import argparse
import os
from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from anomalib.data import MVTec, Folder
from anomalib.deploy import ExportType
import sys
sys.stdout.isatty = lambda: True

from src.config import get_dataset_root, get_results_dir
from src.utils.augmentations import get_train_augmentations

def train_model(model_name, category, dataset_type, img_size, epochs):
    dataset_root = get_dataset_root()
    results_dir = get_results_dir()
    
    output_dir = results_dir / model_name / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Entraînement de {model_name.upper()} sur la catégorie {category} ---")
    print(f"Dataset root : {dataset_root}")
    print(f"Dossier de sortie : {output_dir}")
    
    # 1. Dataset
    transforms = get_train_augmentations(img_size)
    if dataset_type == "mvtec":
        datamodule = MVTec(
            root=dataset_root,
            category=category,
            image_size=(img_size, img_size),
            train_batch_size=32,
            eval_batch_size=32,
            transform=transforms
        )
    else:
        # Folder dataset expects specific structure
        datamodule = Folder(
            name=category,
            root=dataset_root / category,
            normal_dir="train/good",
            abnormal_dir="test/defect", # placeholder for anomalib API
            normal_test_dir="test/good",
            image_size=(img_size, img_size),
            train_batch_size=32,
            eval_batch_size=32,
            transform=transforms
        )
        
    # 2. Modèle
    if model_name == "padim":
        model = Padim()
    elif model_name == "patchcore":
        model = Patchcore()
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
        
    # 3. Engine (Trainer)
    engine = Engine(
        default_root_dir=output_dir,
        max_epochs=epochs,
        task="segmentation",
        accelerator="auto"
    )
    
    # 4. Entraînement
    engine.fit(model=model, datamodule=datamodule)
    
    # 5. Exportation ONNX
    print("--- Exportation au format ONNX ---")
    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=output_dir,
        datamodule=datamodule
    )
    
    print(f"--- Entraînement et Exportation terminés pour {model_name.upper()} ! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement unifié pour PaDiM et PatchCore")
    parser.add_argument("--model", type=str, choices=["padim", "patchcore", "all"], required=True, help="Choix du modèle")
    parser.add_argument("--category", type=str, default="capsule", help="Catégorie du dataset")
    parser.add_argument("--dataset_type", type=str, choices=["mvtec", "folder"], default="mvtec", help="Format du dataset")
    parser.add_argument("--img_size", type=int, default=256, help="Taille des images")
    parser.add_argument("--epochs", type=int, default=1, help="Nombre d'époques")
    
    args = parser.parse_args()
    
    models_to_train = ["padim", "patchcore"] if args.model == "all" else [args.model]
    
    for m in models_to_train:
        train_model(m, args.category, args.dataset_type, args.img_size, args.epochs)
