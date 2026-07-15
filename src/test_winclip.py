import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from anomalib.models import WinClip
from PIL import Image
from pathlib import Path
import numpy as np
import csv

# --- CONFIGURATION ---
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

RACINE_DATASET = Path(__file__).parent.parent / "mvtec_anomaly_detection"
BATCH_SIZE = 1 # Flux industriel : 1 image à la fois
CHEMIN_CLE_USB = Path("/media/barthou/writable")
FICHIER_EXCEL = CHEMIN_CLE_USB / "resultats_inference.csv"

print("Préparation du GPU/CPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Appareil actif : {device} ***")

# Vérification de la clé USB
if not CHEMIN_CLE_USB.exists():
    print(f"[Attention] La clé USB {CHEMIN_CLE_USB} n'est pas montée ! Le fichier sera sauvegardé localement.")
    FICHIER_EXCEL = Path(__file__).parent.parent / "resultats_inference.csv"

# 1. TRANSFORMATION
transform = T.Compose([
    T.Resize((240, 240), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# 2. CRÉATION D'UN DATASET PYTORCH (Sans sauvegarde image)
class MVTecDataset(Dataset):
    def __init__(self, dossier_test):
        self.images_paths = list(dossier_test.glob("**/*.png")) + list(dossier_test.glob("**/*.jpg"))
        
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        img_origine = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_origine)
        return img_tensor, str(img_path)

# 3. FONCTION D'INFÉRENCE CLASSIFICATION PURE AVEC EXPORT EXCEL (CSV)
def evaluer_classification_optimisee(categorie, dossier_test, model, csv_writer):
    dataset = MVTecDataset(dossier_test)
    
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
    
    temps_total_calcul = 0

    with torch.no_grad():
        for batch_tensors, batch_paths in dataloader:
            
            # Conversion du lot en FP16 natif (Tensor Cores)
            if device.type == 'cuda':
                batch_tensors = batch_tensors.half()
            
            batch_tensors = batch_tensors.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize() 
                
            debut_calcul = time.time()
            
            # Inférence
            outputs = model(batch_tensors)
            
            if device.type == 'cuda':
                torch.cuda.synchronize() 
                
            temps_calcul = time.time() - debut_calcul
            temps_total_calcul += temps_calcul
            
            # Extraction du score d'anomalie
            if isinstance(outputs, tuple):
                score = outputs[1].item() if len(outputs) > 1 and outputs[1].numel() == 1 else 0.0
            elif hasattr(outputs, "pred_score"):
                score = outputs.pred_score.item()
            else:
                score = 0.0 # Cas par défaut si format inconnu
                
            # Sauvegarde dans le fichier Excel (CSV)
            nom_image = Path(batch_paths[0]).name
            statut = "NOK (Défaut)" if score > 0.5 else "OK (Normal)" # Seuil arbitraire de 0.5 pour l'exemple
            
            csv_writer.writerow([categorie, nom_image, f"{score:.4f}", statut, f"{temps_calcul*1000:.1f}"])

    nb_images = len(dataset)
    fps = nb_images / temps_total_calcul if temps_total_calcul > 0 else 0
    latence_ms = (temps_total_calcul / nb_images) * 1000 if nb_images > 0 else 0

    print(f"      -> [Chrono] Temps d'inférence pur (GPU) : {temps_total_calcul:.2f} s pour {nb_images} images")
    print(f"      -> [Perf] Latence : {latence_ms:.1f} ms / image  |  Débit : {fps:.1f} FPS")


# --- BOUCLE PRINCIPALE ---
print("\n=== DÉBUT DE LA CLASSIFICATION ZERO-SHOT (FULL OPTIMIZED) ===")
print(f"Les résultats seront sauvegardés dans : {FICHIER_EXCEL}")

categories_trouvees = [d.name for d in RACINE_DATASET.iterdir() if d.is_dir() and d.name in MVTEC_CATEGORIES]

if not categories_trouvees:
    print("Aucun dossier de catégorie MVTec valide trouvé à la racine.")
else:
    # Tentative d'écriture sur la clé USB (ou fallback en local si refus d'accès)
    try:
        f = open(FICHIER_EXCEL, mode='w', newline='', encoding='utf-8')
    except PermissionError:
        print(f"\n[Erreur] Permission refusée pour écrire sur la clé USB ({FICHIER_EXCEL}).")
        FICHIER_EXCEL = Path("/home/barthou/Desktop/resultats_inference.csv")
        print(f"[Fallback] Le fichier sera sauvegardé sur le bureau : {FICHIER_EXCEL}")
        f = open(FICHIER_EXCEL, mode='w', newline='', encoding='utf-8')

    with f:
        writer = csv.writer(f, delimiter=';') # Séparateur Point-Virgule idéal pour Excel français
        # En-têtes des colonnes
        writer.writerow(['Catégorie', 'Image', 'Score Anomalie', 'Statut', 'Latence (ms)'])
        
        for cat in categories_trouvees:
            print(f"\n>>> [TRAITEMENT] Catégorie : {cat.upper()} <<<")
            
            # Désactiver les fenêtres glissantes (scales=tuple())
            try:
                model = WinClip(class_name=cat, scales=tuple()) 
            except Exception:
                model = WinClip(class_name=cat)
                
            model.to(device)
            model.setup("predict")
            model.eval()
            
            # Forcer le modèle en vrai FP16
            if device.type == 'cuda':
                model = model.half()
            
            dossier_test_cat = RACINE_DATASET / cat / "test"
            
            if dossier_test_cat.exists():
                evaluer_classification_optimisee(cat, dossier_test_cat, model, writer)
            else:
                print(f"-> [Erreur] Le sous-dossier 'test' est introuvable pour {cat}.")

print(f"\n=== TERMINÉ ! Résultats disponibles dans {FICHIER_EXCEL} ===")
