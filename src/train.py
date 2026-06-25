import os
import torch
from anomalib.data import MVTec
from anomalib.models import EfficientAd
from anomalib.engine import Engine

def check_gpu():
    print("="*50)
    print("VÉRIFICATION DU MATÉRIEL GPU")
    print("="*50)
    if torch.cuda.is_available():
        print(f"✅ CUDA est disponible !")
        print(f"🔥 GPU détecté : {torch.cuda.get_device_name(0)}")
        print(f"📟 Version de CUDA : {torch.version.cuda}")
        device = "gpu"
    else:
        print("⚠️ Aucun GPU détecté. L'entraînement se fera sur CPU.")
        device = "cpu"
    print("="*50 + "\n")
    return device

def main():
    device = check_gpu()
    category = "bottle"
    dataset_root = os.path.abspath("./mvtec_anomaly_detection")

    print(f"📦 Chargement du dataset MVTec (v1.1.1) pour la catégorie : {category}")
    
    # L'API stable v1.1.1 utilise MVTec et gère parfaitement le paramètre task
    datamodule = MVTec(
        root=dataset_root,
        category=category,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=4,
        task="segmentation"
    )

    print(f"🤖 Initialisation du modèle EfficientAD...")
    model = EfficientAd()

    engine = Engine(
        accelerator=device,
        devices=1,
        max_epochs=10, 
        default_root_dir=f"./results/efficientad/{category}"
    )

    print(f"🚀 Début de l'entraînement d'EfficientAD sur '{category}'...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("✅ Entraînement terminé ! Les résultats sont sauvegardés dans ./results")

if __name__ == "__main__":
    main()
