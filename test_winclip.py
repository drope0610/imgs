import torch
from anomalib.models import WinClip
from PIL import Image
import torchvision.transforms as T
import time
from pathlib import Path

# --- CONFIGURATION ---
DOSSIER_TEST = "Images/imgs/capsule/test" # Ton dossier racine
SEUIL_ANOMALIE = 0.33 # À ajuster selon les premiers résultats

print("Chargement du modèle et préparation du GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# On ajuste le mot clé au nouvel objet ("capsule" ou "pill" fonctionnent bien)
model = WinClip(class_name="capsule") 
model.to(device)
model.setup("predict")
model.eval()

transform = T.Compose([
    T.Resize((240, 240)),
    T.ToTensor(),
])

# --- FONCTION D'ÉVALUATION ---
def evaluer_dossier(chemin_dossier, est_anomalie_reelle):
    # Cherche les formats d'image les plus courants
    images_paths = list(Path(chemin_dossier).glob("*.jpg")) + list(Path(chemin_dossier).glob("*.png"))
    
    if not images_paths:
        return 0, 0, 0.0

    bonnes_predictions = 0
    temps_total = 0.0
    
    for img_path in images_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Warmup GPU
        if temps_total == 0.0:
             with torch.no_grad(): model(img_tensor)

        start_time = time.time()
        with torch.no_grad():
            output = model(img_tensor)
        temps_inference = time.time() - start_time
        temps_total += temps_inference
        
        score = output.pred_score.item()
        prediction_IA = score > SEUIL_ANOMALIE
        
        if prediction_IA == est_anomalie_reelle:
            bonnes_predictions += 1
            
    return bonnes_predictions, len(images_paths), temps_total

# --- LANCEMENT DU BENCHMARK AUTOMATIQUE ---
chemin_racine = Path(DOSSIER_TEST)

# Récupère uniquement les sous-dossiers
dossiers = [d for d in chemin_racine.iterdir() if d.is_dir()]

resultats_detailles = {}
total_images_global = 0
total_bonnes_pred_global = 0
temps_total_global = 0.0

print("\n=== DÉBUT DU BENCHMARK AVANCÉ ===")

for dossier in dossiers:
    nom_categorie = dossier.name
    # Règle logique : Tout ce qui n'est pas strictement nommé "good" est un défaut
    est_anomalie = (nom_categorie != "good") 
    
    print(f"--- Évaluation de la catégorie : [{nom_categorie.upper()}] ---")
    
    bons, total, temps = evaluer_dossier(dossier, est_anomalie_reelle=est_anomalie)
    
    if total > 0:
        precision = (bons / total) * 100
        print(f"-> Précision : {precision:.1f}% ({bons}/{total})")
        
        resultats_detailles[nom_categorie] = {"bons": bons, "total": total}
        total_images_global += total
        total_bonnes_pred_global += bons
        temps_total_global += temps
    else:
        print("-> Dossier vide, ignoré.")

# --- RAPPORTS GLOBAUX ---
if total_images_global > 0:
    temps_moyen_ms = (temps_total_global / total_images_global) * 1000
    precision_globale = (total_bonnes_pred_global / total_images_global) * 100
    
    print("\n================ BILAN FINAL ================")
    print(f"Précision globale : {precision_globale:.1f}% ({total_bonnes_pred_global}/{total_images_global})")
    
    print("\nDétail par type (Matrice de Confusion simplifiée) :")
    for cat, stats in resultats_detailles.items():
        taux = (stats['bons'] / stats['total']) * 100
        if cat == "good":
            print(f"  - Pièces normales bien validées (Vrais Négatifs) : {taux:.1f}% ({stats['bons']}/{stats['total']})")
        else:
            print(f"  - Défaut '{cat}' bien détecté (Vrais Positifs) : {taux:.1f}% ({stats['bons']}/{stats['total']})")
            
    print(f"\nVitesse moyenne d'inférence : {temps_moyen_ms:.2f} ms / image")
    print("=============================================")
else:
    print("\nAucune image n'a été trouvée dans les sous-dossiers. Vérifie le chemin DOSSIER_TEST.")
