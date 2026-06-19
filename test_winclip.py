import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from anomalib.models import WinClip
from PIL import Image
from pathlib import Path
import tifffile as tiff
import numpy as np

# --- CONFIGURATION ---
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

RACINE_DATASET = Path(__file__).parent 
BATCH_SIZE = 8 # Augmente à 16 ou 32 si ta carte graphique a beaucoup de mémoire (VRAM)

print("Préparation du GPU/CPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. TRANSFORMATION CORRIGÉE (Crucial pour la précision de WinCLIP)
transform = T.Compose([
    T.Resize((240, 240), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], # Valeurs officielles CLIP
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# 2. CRÉATION D'UN DATASET PYTORCH (Pour gérer les envois en lots)
class MVTecDataset(Dataset):
    def __init__(self, dossier_test):
        self.images_paths = list(dossier_test.glob("**/*.png")) + list(dossier_test.glob("**/*.jpg"))
        
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        img_origine = Image.open(img_path).convert("RGB")
        largeur, hauteur = img_origine.size
        
        img_tensor = transform(img_origine)
        
        # On retourne le tenseur, le chemin (pour la sauvegarde) et la taille d'origine
        return img_tensor, str(img_path), largeur, hauteur


# 3. FONCTION D'INFÉRENCE OPTIMISÉE
def generer_anomaly_maps_optimise(dossier_test, chemin_sortie_racine, model):
    dataset = MVTecDataset(dossier_test)
    
    if len(dataset) == 0:
        return

    # Le DataLoader s'occupe de grouper les images par lots (Batch)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
    
    with torch.no_grad(): # Désactive le calcul des gradients (accélère l'inférence)
        for batch_tensors, batch_paths, batch_largeurs, batch_hauteurs in dataloader:
            
            # Envoi du lot complet au GPU
            batch_tensors = batch_tensors.to(device)
            
            # Le GPU calcule les anomalies pour TOUTES les images du lot en même temps
            outputs = model(batch_tensors)
            
            # Récupération des cartes d'anomalies (Taille : Batch x 1 x 240 x 240)
            anomaly_maps = outputs.anomaly_map 
            
            # --- TRAITEMENT DU LOT ---
            for i in range(len(batch_paths)):
                img_path = Path(batch_paths[i])
                largeur_orig = batch_largeurs[i].item()
                hauteur_orig = batch_hauteurs[i].item()
                
                # 1. Redimensionnement rapide sur GPU via interpolation
                map_tensor = anomaly_maps[i].unsqueeze(0) # Garder le format 1x1xHxW
                map_resized = F.interpolate(map_tensor, size=(hauteur_orig, largeur_orig), mode='bilinear', align_corners=False)
                
                # 2. Conversion finale en Numpy pour la sauvegarde
                anomaly_map_final = map_resized.squeeze().cpu().numpy().astype(np.float32)
                
                # 3. Création des dossiers et sauvegarde
                # (Extrait le nom du défaut qui est le dossier parent de l'image)
                nom_defaut = img_path.parent.name 
                dossier_export = chemin_sortie_racine / nom_defaut
                dossier_export.mkdir(parents=True, exist_ok=True)
                
                nom_fichier_sortie = dossier_export / f"{img_path.stem}.tiff"
                tiff.imwrite(str(nom_fichier_sortie), anomaly_map_final)


# --- BOUCLE PRINCIPALE ---
print("\n=== DÉBUT DE LA GÉNÉRATION DES PRÉDICTIONS GLOBALES ===")

categories_trouvees = [d.name for d in RACINE_DATASET.iterdir() if d.is_dir() and d.name in MVTEC_CATEGORIES]

if not categories_trouvees:
    print("Aucun dossier de catégorie MVTec valide trouvé à la racine.")
else:
    print(f"Catégories détectées et prêtes à être traitées : {categories_trouvees}")
    
    for cat in categories_trouvees:
        print(f"\n>>> [TRAITEMENT] Catégorie : {cat.upper()} <<<")
        
        # Initialisation (le CPE fait sa magie en coulisses)
        model = WinClip(class_name=cat) 
        model.to(device)
        model.setup("predict")
        model.eval() # Très important de s'assurer qu'on est en mode évaluation
        
        dossier_test_cat = RACINE_DATASET / cat / "test"
        dossier_sortie_cat = RACINE_DATASET / "predictions" / cat / "test"
        
        if dossier_test_cat.exists():
            generer_anomaly_maps_optimise(dossier_test_cat, dossier_sortie_cat, model)
            print(f"-> OK : Toutes les cartes .tiff pour '{cat}' ont été générées.")
        else:
            print(f"-> [Erreur] Le sous-dossier 'test' est introuvable pour {cat}.")

print("\n=== TOUTES LES PRÉDICTIONS ONT ÉTÉ GÉNÉRÉES AVEC SUCCÈS ===")
