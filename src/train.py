import os
import torch
# Remplacement de MVTec par MVTecAD suite à la mise à jour de l'API
from anomalib.data import MVTecAD
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
        print("⚠️ Aucun GPU détecté. L'entraînement se fera sur CPU (très lent).")
        device = "cpu"
    print("="*50 + "\n")
    return device

def main():
    # 1. Vérification du matériel
    device = check_gpu()

    # 2. Configuration des chemins
    category = "bottle"
    dataset_root = os.path.abspath("./mvtec_anomaly_detection")

    print(f"📦 Chargement du dataset MVTecAD pour la catégorie : {category}")
    
    # Utilisation de la nouvelle classe MVTecAD
    datamodule = MVTecAD(
        root=dataset_root,
        category=category,
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4
    )

    # 3. Initialisation du modèle (EfficientAD)
    print(f"🤖 Initialisation du modèle EfficientAD...")
    model = EfficientAd()

    # 4. Configuration du moteur d'entraînement
    engine = Engine(
        accelerator=device,
        devices=1,
        max_epochs=10, 
        default_root_dir=f"./results/efficientad/{category}"
    )

    # 5. Lancement de l'entraînement
    print(f"🚀 Début de l'entraînement d'EfficientAD sur '{category}'...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("✅ Entraînement terminé ! Les résultats et checkpoints sont sauvegardés dans le dossier ./results")

if __name__ == "__main__":
    main()
