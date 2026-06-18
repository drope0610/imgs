import torch
from anomalib.models import WinClip
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import tifffile as tiff
import numpy as np

# --- CONFIGURATION ---
CATEGORIE = "capsule"  # Modifier selon l'objet testé
DOSSIER_TEST = f"Images/imgs/{CATEGORIE}/test"  # Ton dossier contenant 'good', 'crack', etc.
DOSSIER_SORTIE_PRED = f"predictions/{CATEGORIE}/test"  # Arborescence requise par MVTec

print("Chargement du modèle WinClip...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WinClip(class_name=CATEGORIE) 
model.to(device)
model.setup("predict")
model.eval()

# Transformation pour WinClip (Input standardisé)
transform = T.Compose([
    T.Resize((240, 240)),
    T.ToTensor(),
])

def generer_anomaly_maps(chemin_dossier_test, chemin_sortie_racine):
    chemin_test = Path(chemin_dossier_test)
    
    # Parcourir chaque sous-dossier de défauts ('good', 'crack', etc.)
    for sous_dossier in chemin_test.iterdir():
        if not sous_dossier.is_dir():
            continue
            
        nom_defaut = sous_dossier.name
        print(f"Traitement du type de défaut : {nom_defaut}")
        
        # Créer le dossier de sortie correspondant requis par MVTec
        dossier_export = Path(chemin_sortie_racine) / nom_defaut
        dossier_export.mkdir(parents=True, exist_ok=True)
        
        # Trouver toutes les images
        images_paths = list(sous_dossier.glob("*.jpg")) + list(sous_dossier.glob("*.png"))
        
        for img_path in images_paths:
            # 1. Charger l'image d'origine pour obtenir ses vraies dimensions
            img_origine = Image.open(img_path).convert("RGB")
            largeur_orig, hauteur_orig = img_origine.size
            
            # 2. Préparer le tenseur pour le modèle
            img_tensor = transform(img_origine).unsqueeze(0).to(device)
            
            # 3. Inférence
            with torch.no_grad():
                output = model(img_tensor)
            
            # 4. Extraire l'anomaly map (carte pixel par pixel)
            # output.anomaly_map a généralement une forme (1, 1, H, W)
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
            
            # 5. Redimensionner la carte à la taille de l'image originale (Requis par MVTec !)
            anomaly_map_resized = Image.fromarray(anomaly_map)
            anomaly_map_resized = anomaly_map_resized.resize((largeur_orig, hauteur_orig), resample=Image.BILINEAR)
            anomaly_map_final = np.array(anomaly_map_resized).astype(np.float32)
            
            # 6. Sauvegarder au format .tiff avec le même nom d'origine
            nom_fichier_sortie = dossier_export / f"{img_path.stem}.tiff"
            tiff.imwrite(str(nom_fichier_sortie), anomaly_map_final)

print("\n=== DÉBUT DE LA GÉNÉRATION DES CARTES D'ANOMALIES ===")
generer_anomaly_maps(DOSSIER_TEST, DOSSIER_SORTIE_PRED)
print(f"\n=== TERMINÉ ! Les prédictions sont prêtes dans : {DOSSIER_SORTIE_PRED} ===")
