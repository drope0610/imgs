import json
import csv
import os

def main():
    metrics_path = './metrics/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"⚠️  Fichier {metrics_path} introuvable. Avez-vous lancé l'évaluation ?")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Récupérer toutes les catégories évaluées
    categories = [k for k in metrics.keys() if not k.startswith('mean_')]
    
    csv_path = './metrics/resultats_benchmark.csv'
    with open(csv_path, 'w', newline='') as f:
        # On utilise le point-virgule pour que l'ouverture dans Excel (en France) soit automatique
        writer = csv.writer(f, delimiter=';') 
        writer.writerow(['Catégorie', 'AU_PRO (Localisation du défaut)', 'AU_ROC (Détection globale)'])
        
        for cat in categories:
            pro = round(metrics[cat].get('au_pro', 0.0), 3)
            roc = round(metrics[cat].get('classification_au_roc', 0.0), 3)
            # Remplacer les points par des virgules si nécessaire pour Excel français, 
            # mais généralement Excel gère bien si on spécifie bien lors de l'import.
            writer.writerow([cat, str(pro).replace('.', ','), str(roc).replace('.', ',')])
            
        # Ajouter la moyenne globale à la fin
        mean_pro = round(metrics.get('mean_au_pro', 0.0), 3)
        mean_roc = round(metrics.get('mean_classification_au_roc', 0.0), 3)
        writer.writerow(['MOYENNE GLOBALE', str(mean_pro).replace('.', ','), str(mean_roc).replace('.', ',')])
        
    print(f"\n📊 [EXCEL] Résultats sauvegardés avec succès dans : {csv_path}")
    print("Vous pouvez double-cliquer dessus pour l'ouvrir directement dans Excel !")

if __name__ == '__main__':
    main()
