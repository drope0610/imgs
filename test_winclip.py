import torch
from anomalib.models import WinClip
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import tifffile as tiff
import numpy as np

# --- LISTE DES 15 CATÉGORIES OFFICIELLES DE MVTEC AD ---
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# Ton dossier racine (où se trouvent capsule, bottle, etc.)
RACINE_DATASET = Path(__file__).parent 

print("Préparation du GPU/CPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation requise pour l'entrée du modèle WinClip
transform = T.Compose([
    T.Resize((240, 240)),
    T.ToTensor(),
])

def generer_anomaly_maps_pour_categorie(chemin_dossier_test, chemin_sortie_racine, model):
    for sous_dossier in chemin_dossier_test.iterdir():
        if not sous_dossier.is_dir():
            continue
            
        nom_defaut = sous_dossier.name
        print(f"   -> Dossier de défaut : {nom_defaut}")
        
        # Créer l'arborescence : predictions/<categorie>/test/<nom_defaut>
        dossier_export = chemin_sortie_racine / nom_defaut
        dossier_export.mkdir(parents=True, exist_ok=True)
        
        images_paths = list(sous_dossier.glob("*.png")) + list(sous_dossier.glob("*.jpg"))
        
        for img_path in images_paths:
            img_origine = Image.open(img_path).convert("RGB")
            largeur_orig, hauteur_orig = img_origine.size
            
            img_tensor = transform(img_origine).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
            
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
            
            # Redimensionnement à la taille d'origine (Exigence MVTec)
            anomaly_map_resized = Image.fromarray(anomaly_map)
            anomaly_map_resized = anomaly_map_resized.resize((largeur_orig, hauteur_orig), resample=Image.BILINEAR)
            anomaly_map_final = np.array(anomaly_map_resized).astype(np.float32)
            
            # Sauvegarde .tiff
            nom_fichier_sortie = dossier_export / f"{img_path.stem}.tiff"
            tiff.imwrite(str(nom_fichier_sortie), anomaly_map_final)

# --- BOUCLE AUTOMATIQUE SUR LES DOSSIERS PRÉSENTS ---
print("\n=== DÉBUT DE LA GÉNÉRATION DES PRÉDICTIONS GLOBALES ===")

# Détection automatique des dossiers MVTec valides dans ton répertoire
categories_trouvees = [d.name for d in RACINE_DATASET.iterdir() if d.is_dir() and d.name in MVTEC_CATEGORIES]

if not categories_trouvees:
    print("Aucun dossier de catégorie MVTec valide trouvé à la racine.")
else:
    print(f"Catégories détectées et prêtes à être traitées : {categories_trouvees}")
    
    for cat in categories_trouvees:
        print(f"\n>>> [TRAITEMENT] Catégorie : {cat.upper()} <<<")
        
        # On réinitialise WinClip avec les prompts textuels de la bonne catégorie
        model = WinClip(class_name=cat) 
        model.to(device)
        model.setup("predict")
        model.eval()
        
        dossier_test_cat = RACINE_DATASET / cat / "test"
        dossier_sortie_cat = RACINE_DATASET / "predictions" / cat / "test"
        
        if dossier_test_cat.exists():
            generer_anomaly_maps_pour_categorie(dossier_test_cat, dossier_sortie_cat, model)
            print(f"-> OK : Toutes les cartes .tiff pour '{cat}' ont été générées.")
        else:
            print(f"-> [Erreur] Le sous-dossier 'test' est introuvable pour {cat}.")

print("\n=== TOUTES LES PRÉDICTIONS ONT ÉTÉ GÉNÉRÉES AVEC SUCCÈS ===")
