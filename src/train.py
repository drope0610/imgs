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
        print("⚠️ Aucun GPU détecté. L'entraînement se fera sur CPU (très lent).")
        device = "cpu"
    print("="*50 + "\n")
    return device

def main():
    # 1. Vérification du matériel
    device = check_gpu()

    # 2. Configuration des chemins (Modifier selon votre catégorie de test, ex: 'bottle')
    category = "bottle"
    dataset_root = os.path.abspath("./mvtec_anomaly_detection")

    print(f"📦 Chargement du dataset MVTec pour la catégorie : {category}")
    
    # Configuration du DataModule Anomalib
    datamodule = MVTec(
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

    # 4. Configuration du moteur d'entraînement (Engine)
    # C'est ici qu'Anomalib gère les boucles de calcul, les checkpoints et le device
    engine = Engine(
        accelerator=device,
        devices=1,
        max_epochs=10,  # Réduit à 10 pour un premier test rapide, EfficientAD converge vite
        default_root_dir=f"./results/efficientad/{category}"
    )

    # 5. Lancement de l'entraînement
    print(f"🚀 Début de l'entraînement d'EfficientAD sur '{category}'...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("✅ Entraînement terminé ! Les résultats et checkpoints sont sauvegardés dans le dossier ./results")

if __name__ == "__main__":
    main()
