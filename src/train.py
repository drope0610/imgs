import os
import argparse
import torch
from anomalib.data import MVTec
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.deploy import ExportType

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
    # Configuration des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement MVTec AD")
    parser.add_argument(
        "--category", 
        type=str, 
        default="bottle", 
        help="Catégorie MVTec à entraîner (ex: cable, capsule, wood...)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Nombre d'époques d'entraînement"
    )
    args = parser.parse_args()

    device = check_gpu()
    category = args.category
    dataset_root = os.path.abspath("./mvtec_anomaly_detection")

    print(f"📦 Chargement du dataset MVTec pour la catégorie : {category}")
    
    datamodule = MVTec(
        root=dataset_root,
        category=category,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=4,
        task="segmentation"
    )

    print(f"🤖 Initialisation du modèle EfficientAD pour '{category}'...")
    model = EfficientAd()

    engine = Engine(
        accelerator=device,
        devices=1,
        max_epochs=args.epochs, 
        default_root_dir=f"./results/efficientad/{category}"
    )

    print(f"🚀 Début de l'entraînement d'EfficientAD sur '{category}' ({args.epochs} époques)...")
    engine.fit(model=model, datamodule=datamodule)
    
    print(f"📦 Exportation du modèle '{category}' au format ONNX...")
    engine.export(model=model, export_type=ExportType.ONNX)

    print(f"✅ [SUCCÈS] Entraînement et exportation ONNX terminés pour '{category}' !")

if __name__ == "__main__":
    main()
