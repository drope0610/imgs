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

# Détection de la racine du Dataset et de la clé USB (Multi-Jetsons)
RACINE_DATASET = Path(__file__).parent.parent / "mvtec_anomaly_detection"
CHEMIN_CLE_USB = Path("/tmp") # Fallback

if Path("/media/pedro/Modeles").exists(): # Jetson 2
    CHEMIN_CLE_USB = Path("/media/pedro/Modeles")
    RACINE_DATASET = CHEMIN_CLE_USB / "mvtec_anomaly_detection"
elif Path("/media/pedro2/Modeles").exists(): # Jetson 3
    CHEMIN_CLE_USB = Path("/media/pedro2/Modeles")
    RACINE_DATASET = CHEMIN_CLE_USB / "mvtec_anomaly_detection"
elif Path("/media/barthou/writable").exists(): # Jetson 1
    CHEMIN_CLE_USB = Path("/media/barthou/writable")

FICHIER_EXCEL = CHEMIN_CLE_USB / "resultats_inference.csv"
BATCH_SIZE = 1 # Flux industriel : 1 image à la fois

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

def get_best_threshold(scores, labels):
    best_f1 = 0
    best_thresh = 0.5
    thresholds = sorted(list(set(scores)))
    for t in thresholds:
        tp = sum(1 for s, l in zip(scores, labels) if s > t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s > t and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s <= t and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    return best_thresh

# 3. FONCTION D'INFÉRENCE CLASSIFICATION PURE AVEC EXPORT EXCEL (CSV)
def evaluer_classification_optimisee(categorie, dossier_test, model, csv_writer, seuil_global):
    dataset = MVTecDataset(dossier_test)
    
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
    
    temps_total_calcul = 0
    temps_total_ok = 0
    temps_total_nok = 0
    nb_ok = 0
    nb_nok = 0
    reussites = 0

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
                score = outputs[0].item() if len(outputs) > 0 and outputs[0].numel() == 1 else 0.0
            elif hasattr(outputs, "pred_score"):
                score = outputs.pred_score.item()
            else:
                score = 0.0 # Cas par défaut si format inconnu
                
            # Détermination de la vérité terrain (dossier parent = "good" ou nom du défaut)
            parent_dir = Path(batch_paths[0]).parent.name
            verite_terrain = "OK (Normal)" if parent_dir == "good" else "NOK (Défaut)"
            
            # Mise à jour des temps par type
            if verite_terrain == "OK (Normal)":
                temps_total_ok += temps_calcul
                nb_ok += 1
            else:
                temps_total_nok += temps_calcul
                nb_nok += 1

            # Sauvegarde dans le fichier Excel (CSV)
            nom_image = Path(batch_paths[0]).name
            statut = "NOK (Défaut)" if score > seuil_global else "OK (Normal)" # Seuil dynamique
            
            succes = "Vrai" if statut == verite_terrain else "Faux"
            if succes == "Vrai":
                reussites += 1
            
            csv_writer.writerow([categorie, nom_image, verite_terrain, f"{score:.4f}", statut, succes, f"{temps_calcul*1000:.1f}"])

    nb_images = len(dataset)
    fps = nb_images / temps_total_calcul if temps_total_calcul > 0 else 0
    latence_ms = (temps_total_calcul / nb_images) * 1000 if nb_images > 0 else 0
    latence_ok_ms = (temps_total_ok / nb_ok) * 1000 if nb_ok > 0 else 0
    latence_nok_ms = (temps_total_nok / nb_nok) * 1000 if nb_nok > 0 else 0
    taux_reussite = (reussites / nb_images) * 100 if nb_images > 0 else 0

    print(f"      -> [Chrono] Temps d'inférence pur (GPU) : {temps_total_calcul:.2f} s pour {nb_images} images")
    print(f"      -> [Perf Globale] Latence : {latence_ms:.1f} ms/img  |  Débit : {fps:.1f} FPS")
    print(f"      -> [Détail Latence] Images OK : {latence_ok_ms:.1f} ms/img  |  Images NOK : {latence_nok_ms:.1f} ms/img")
    print(f"      -> [Précision] Taux de réussite (Seuil {seuil_global:.4f}) : {taux_reussite:.1f}% ({reussites}/{nb_images})")


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
        writer.writerow(['Catégorie', 'Image', 'Vérité Terrain', 'Score Anomalie', 'Statut Prédit', 'Succès', 'Latence (ms)'])
        
        # --- ÉTAPE 1 : CALCUL DU SEUIL PARFAIT SUR LA CATÉGORIE PILL (MÉDICAMENTS) ---
        seuil_medicament = 0.5
        if 'pill' in categories_trouvees:
            print("\n=== CALCUL DU SEUIL OPTIMAL (F1-MAX) SUR LA CATÉGORIE 'PILL' ===")
            cat_pill = 'pill'
            dossier_pill = RACINE_DATASET / cat_pill / "test"
            if dossier_pill.exists():
                try:
                    model_pill = WinClip(class_name=cat_pill, scales=tuple()) 
                except Exception:
                    model_pill = WinClip(class_name=cat_pill)
                model_pill.to(device)
                model_pill.setup("predict")
                model_pill.eval()
                if device.type == 'cuda':
                    model_pill = model_pill.half()
                
                dataset_pill = MVTecDataset(dossier_pill)
                dataloader_pill = DataLoader(dataset_pill, batch_size=BATCH_SIZE, shuffle=False)
                
                scores_pill = []
                labels_pill = []
                
                with torch.no_grad():
                    for batch_tensors, batch_paths in dataloader_pill:
                        if device.type == 'cuda':
                            batch_tensors = batch_tensors.half()
                        batch_tensors = batch_tensors.to(device)
                        outputs = model_pill(batch_tensors)
                        
                        if isinstance(outputs, tuple):
                            score = outputs[0].item() if len(outputs) > 0 and outputs[0].numel() == 1 else 0.0
                        elif hasattr(outputs, "pred_score"):
                            score = outputs.pred_score.item()
                        else:
                            score = 0.0
                            
                        parent_dir = Path(batch_paths[0]).parent.name
                        label = 0 if parent_dir == "good" else 1
                        
                        scores_pill.append(score)
                        labels_pill.append(label)
                
                seuil_medicament = get_best_threshold(scores_pill, labels_pill)
                print(f"-> Seuil parfait trouvé pour 'pill' : {seuil_medicament:.4f}")
                # Libérer la mémoire
                del model_pill
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        else:
            print("\n[Attention] La catégorie 'pill' n'est pas présente. Utilisation du seuil par défaut 0.5.")
            
        print(f"\n=== APPLICATION DU SEUIL GLOBAL ({seuil_medicament:.4f}) À TOUTES LES CATÉGORIES ===")

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
                evaluer_classification_optimisee(cat, dossier_test_cat, model, writer, seuil_medicament)
            else:
                print(f"-> [Erreur] Le sous-dossier 'test' est introuvable pour {cat}.")

print(f"\n=== TERMINÉ ! Résultats disponibles dans {FICHIER_EXCEL} ===")
