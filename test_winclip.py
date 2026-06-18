import torch
from anomalib.models import WinClip
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import tifffile as tiff
import numpy as np

# --- CONFIGURATION AUTOMATIQUE VIA TON ARBORESCENCE ---
CATEGORIE = "capsule"
# Pointe directement vers ./capsule/test
DOSSIER_TEST = Path(__file__).parent / CATEGORIE / "test"  
# Générera l'arborescence requise : ./predictions/capsule/test
DOSSIER_SORTIE_PRED = Path(__file__).parent / "predictions" / CATEGORIE / "test"  

print(f"Chargement du modèle WinClip pour la catégorie : {CATEGORIE}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WinClip(class_name=CATEGORIE) 
model.to(device)
model.setup("predict")
model.eval()

# Transformation requise pour l'entrée du modèle WinClip
transform = T.Compose([
    T.Resize((240, 240)),
    T.ToTensor(),
])

def generer_anomaly_maps(chemin_dossier_test, chemin_sortie_racine):
    # Parcourir chaque sous-dossier de défauts ('good', 'crack', 'poke', etc.)
    for sous_dossier in chemin_dossier_test.iterdir():
        if not sous_dossier.is_dir():
            continue
            
        nom_defaut = sous_dossier.name
        print(f"-> Génération des cartes pour le défaut : {nom_defaut}")
        
        # Créer le dossier de sortie requis : predictions/capsule/test/<nom_defaut>
        dossier_export = chemin_sortie_racine / nom_defaut
        dossier_export.mkdir(parents=True, exist_ok=True)
        
        # Récupère toutes les images .png du dossier actuel
        images_paths = list(sous_dossier.glob("*.png"))
        
        for img_path in images_paths:
            # 1. Charger l'image originale pour connaître sa résolution native
            img_origine = Image.open(img_path).convert("RGB")
            largeur_orig, hauteur_orig = img_origine.size
            
            # 2. Convertir pour le modèle
            img_tensor = transform(img_origine).unsqueeze(0).to(device)
            
            # 3. Inférence
            with torch.no_grad():
                output = model(img_tensor)
            
            # 4. Extraction de la carte d'anomalie pixel par pixel
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
            
            # 5. Redimensionnement strict à la taille d'origine (Exigence MVTec)
            anomaly_map_resized = Image.fromarray(anomaly_map)
            anomaly_map_resized = anomaly_map_resized.resize((largeur_orig, hauteur_orig), resample=Image.BILINEAR)
            anomaly_map_final = np.array(anomaly_map_resized).astype(np.float32)
            
            # 6. Sauvegarde au format .tiff (porte le même nom que l'image ex: 000.tiff)
            nom_fichier_sortie = dossier_export / f"{img_path.stem}.tiff"
            tiff.imwrite(str(nom_fichier_sortie), anomaly_map_final)

print("\n=== DÉBUT DE LA GÉNÉRATION DES IMAGES TIFF REQUISES ===")
generer_anomaly_maps(DOSSIER_TEST, DOSSIER_SORTIE_PRED)
print(f"\n=== TERMINÉ ! Les fichiers de prédiction sont dans : {DOSSIER_SORTIE_PRED} ===")
