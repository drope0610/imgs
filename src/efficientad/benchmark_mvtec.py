import os
import pandas as pd
from pathlib import Path
from anomalib.data import MVTec
from anomalib.models import EfficientAd
from anomalib.engine import Engine

def main():
    dataset_path = Path("./mvtec_anomaly_detection")
    output_csv = Path("./results_efficientad/mvtec_benchmark_results.csv")
    
    # 1. Lister automatiquement toutes les catégories (les sous-dossiers)
    categories = [f.name for f in dataset_path.iterdir() if f.is_dir()]
    print(f"Catégories trouvées ({len(categories)}) : {categories}")
    
    all_results = []

    # 2. Boucler sur chaque catégorie
    for category in categories:
        print("\n" + "="*50)
        print(f" Lancement du Benchmark pour : {category.upper()}")
        print("="*50)
        
        try:
            # Configuration du dataset pour la catégorie courante
            datamodule = MVTec(
                root=dataset_path,
                category=category,
                image_size=(256, 256),
                train_batch_size=32,
                eval_batch_size=32,
            )
            datamodule.setup()

            # Initialisation d'EfficientAD
            model = EfficientAd(model_size="S")

            # Configuration du moteur d'entraînement
            engine = Engine(
                accelerator="auto",
                max_epochs=10, # Converge en 10 époques sur MVTec
                default_root_dir=f"./results_efficientad/{category}",
                metrics={"image": ["AUROC"], "pixel": ["AUROC"]} # Métriques standards de l'article
            )

            # Entraînement
            engine.fit(model=model, datamodule=datamodule)

            # Évaluation
            metrics = engine.test(model=model, datamodule=datamodule)
            
            # Extraction des scores (les clés exactes peuvent varier selon la version d'Anomalib, souvent 'image_AUROC')
            res = {"category": category}
            for k, v in metrics[0].items():
                res[k] = float(v)
            
            all_results.append(res)
            print(f"Résultats pour {category} : {res}")

        except Exception as e:
            print(f"❌ Erreur lors du traitement de la catégorie {category} : {e}")
            continue

    # 3. Générer et sauvegarder le rapport global
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Calculer la moyenne globale (comme dans l'article de recherche)
        mean_row = df.mean(numeric_only=True).to_dict()
        mean_row["category"] = "Mean"
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        
        # Sauvegarde
        os.makedirs(output_csv.parent, exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        print("\n" + "═"*50)
        print(f" Benchmark terminé ! Tableau des scores sauvegardé dans : {output_csv}")
        print("═"*50)
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
