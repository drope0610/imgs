from anomalib.data import MVTec
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from pathlib import Path

def main():
    # Path absolu ou relatif vers votre dossier MVTec
    dataset_path = Path("./mvtec_anomaly_detection")
    
    # 1. Configuration du DataModule spécifique à MVTec
    # Nous testons ici sur la catégorie 'bottle'
    datamodule = MVTec(
        root=dataset_path,
        category="bottle",
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
    )
    datamodule.setup()

    # 2. Initialisation d'EfficientAD (Taille S pour aller vite)
    model = EfficientAd(model_size="S")

    # 3. Configuration de l'Engine
    engine = Engine(
        accelerator="auto",     # Détecte automatiquement votre GPU ou CPU
        max_epochs=10,          # EfficientAD converge très rapidement
        default_root_dir="./results_efficientad"
    )

    # 4. Lancement de l'entraînement
    print("Création du modèle et début de l'entraînement sur 'bottle'...")
    engine.fit(model=model, datamodule=datamodule)

    # 5. Évaluation des performances (Calcul de l'AUROC, etc.)
    print("Début des tests et calcul des métriques...")
    metrics = engine.test(model=model, datamodule=datamodule)
    print("\n--- RÉSULTATS FINAUX ---")
    print(metrics)

if __name__ == "__main__":
    main()
