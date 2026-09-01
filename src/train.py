import os
import argparse
import shutil
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_dataset_root, get_results_dir
from src.utils.augmentations import get_train_augmentations

import torch
from anomalib.data import MVTec, Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.deploy import ExportType

import rich.console
rich.console.Console.clear_live = lambda self: None

if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.ModuleType("distributed")
if not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False

def check_gpu():
    print("="*50)
    print("VÉRIFICATION DU MATÉRIEL GPU")
    print("="*50)
    if torch.cuda.is_available():
        print(f"✅ CUDA est disponible !")
        print(f"🔥 GPU détecté : {torch.cuda.get_device_name(0)}")
        device = "gpu"
    else:
        print("⚠️ Aucun GPU détecté. L'entraînement se fera sur CPU.")
        device = "cpu"
    print("="*50 + "\n")
    return device

def main():
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement Unifié")
    parser.add_argument("--category", type=str, default="capsule", help="Catégorie (ex: capsule, pillqc)")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument("--dataset_type", type=str, default="mvtec", choices=["mvtec", "folder"], help="Format du dataset")
    parser.add_argument("--img_size", type=int, default=256, help="Taille des images d'entraînement")
    parser.add_argument("--run_name", type=str, default="", help="Nom du dossier de sauvegarde (optionnel)")
    args = parser.parse_args()

    device = check_gpu()
    category = args.category
    
    print(f"📦 Chargement du dataset pour la catégorie : {category}")
    
    if args.dataset_type == "mvtec":
        dataset_root = get_dataset_root()
        datamodule = MVTec(
            root=str(dataset_root),
            category=category,
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=4,
            image_size=(args.img_size, args.img_size),
            task="segmentation"
        )
    elif args.dataset_type == "folder":
        # Specific logic for pillqc style structure
        dataset_root_resolved = Path.home() / "Desktop" / "Images" / "imgs" / "datasets" / category / "images"
        if not dataset_root_resolved.exists():
            raise FileNotFoundError(f"Dossier dataset introuvable : {dataset_root_resolved}")
        
        datamodule = Folder(
            name=category,
            root=dataset_root_resolved,
            normal_dir="normal",
            abnormal_dir="dirt", # stub for anomalib structure
            image_size=(args.img_size, args.img_size),
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=4,
            task="classification"
        )

    print(f"🤖 Initialisation du modèle EfficientAD pour '{category}'...")
    model = EfficientAd()

    # Apply Custom Data Augmentation to the training dataset
    datamodule.setup() 
    if hasattr(datamodule, "train_data") and datamodule.train_data is not None:
        print("🪄 Injection des transformations custom (Data Augmentation : Inpainting & Brightness) sur le jeu d'entraînement...")
        # Override the transform function
        datamodule.train_data.transform = get_train_augmentations(args.img_size)
    else:
        print("⚠️ Impossible d'injecter la Data Augmentation, `train_data` non trouvé.")

    results_dir = get_results_dir() / "efficientad" / category
    if args.run_name:
        results_dir = results_dir / args.run_name
        
    engine = Engine(
        accelerator=device,
        devices=1,
        max_epochs=args.epochs, 
        default_root_dir=str(results_dir)
    )

    print(f"🚀 Début de l'entraînement d'EfficientAD sur '{category}' ({args.epochs} époques)...")
    engine.fit(model=model, datamodule=datamodule)
    
    print(f"📦 Exportation du modèle '{category}' au format ONNX...")
    engine.export(model=model, export_type=ExportType.ONNX)
    
    print(f"✅ [SUCCÈS] Entraînement et exportation ONNX terminés pour '{category}' !")

if __name__ == "__main__":
    main()
