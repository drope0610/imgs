import json
import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./metrics/', help='Dossier où sauvegarder le CSV')
    parser.add_argument('--filename', default='resultats_benchmark.csv', help='Nom du fichier CSV final')
    args = parser.parse_args()

    metrics_path = './metrics/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"⚠️  Fichier {metrics_path} introuvable. Avez-vous lancé l'évaluation ?")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Récupérer toutes les catégories évaluées
    categories = [k for k in metrics.keys() if not k.startswith('mean_')]
    
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.filename)
    
    with open(csv_path, 'w', newline='') as f:
        # On utilise le point-virgule pour que l'ouverture dans Excel (en France) soit automatique
        writer = csv.writer(f, delimiter=';') 
        writer.writerow(['Catégorie', 'AU_PRO (Localisation du défaut)', 'AU_ROC (Détection globale)'])
        
        for cat in categories:
            pro = round(metrics[cat].get('au_pro', 0.0), 3)
            roc = round(metrics[cat].get('classification_au_roc', 0.0), 3)
            writer.writerow([cat, str(pro).replace('.', ','), str(roc).replace('.', ',')])
            
        # Ajouter la moyenne globale à la fin
        mean_pro = round(metrics.get('mean_au_pro', 0.0), 3)
        mean_roc = round(metrics.get('mean_classification_au_roc', 0.0), 3)
        writer.writerow(['MOYENNE GLOBALE', str(mean_pro).replace('.', ','), str(mean_roc).replace('.', ',')])
        
    print(f"\n📊 [EXCEL] Résultats sauvegardés avec succès dans : {csv_path}")

if __name__ == '__main__':
    main()
