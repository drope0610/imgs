import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from anomalib.models import Padim, Patchcore
from src.utils.metrics import get_best_threshold

def apply_heatmap(image_np, anomaly_map_np):
    """
    Superpose la heatmap sur l'image d'origine.
    """
    # Normaliser la carte d'anomalie entre 0 et 255
    anomaly_map_norm = cv2.normalize(anomaly_map_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Appliquer la colormap JET
    heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
    
    # Superposer avec l'image originale (50% / 50%)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
    
    return overlay

def test_model(model_name, category, checkpoint_path, output_dir):
    print(f"--- Évaluation de {model_name.upper()} sur la catégorie {category} ---")
    
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*** Appareil actif : {device} ***")
    
    out_dir = Path(output_dir) if output_dir else Path(f"results/eval_{model_name}_{category}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Chargement du modèle
    if model_name == "padim":
        model = Padim.load_from_checkpoint(checkpoint_path)
    elif model_name == "patchcore":
        model = Patchcore.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
        
    model.to(device)
    model.eval()
    
    # 3. Préparation des images
    from src.config import get_dataset_root
    dataset_root = get_dataset_root()
    test_dir = dataset_root / category / "test"
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Dossier de test introuvable : {test_dir}")
        
    img_size = 256 # Taille par défaut
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    
    scores = []
    labels = []
    
    total_time = 0.0
    num_images = 0
    
    # 4. Inférence boucle manuelle
    with torch.no_grad():
        for defect_type in os.listdir(test_dir):
            defect_path = test_dir / defect_type
            if not defect_path.is_dir():
                continue
                
            label = 0 if defect_type == "good" else 1
            
            for img_name in os.listdir(defect_path):
                if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = defect_path / img_name
                img_pil = Image.open(img_path).convert("RGB")
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                # Mesure de performance
                start_time = time.perf_counter()
                
                output = model(img_tensor)
                
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
                num_images += 1
                
                # Traitement de la sortie PyTorch
                if isinstance(output, tuple) or isinstance(output, list):
                    anomaly_map = output[0]
                    score = output[1].item() if len(output) > 1 else anomaly_map.max().item()
                elif isinstance(output, dict):
                    anomaly_map = output.get("anomaly_map", output.get("pred_masks"))
                    score = output.get("pred_scores")
                    if score is None and anomaly_map is not None:
                        score = anomaly_map.max().item()
                    elif score is not None:
                        score = score.item()
                    else:
                        raise ValueError("Le modèle ne renvoie ni anomaly_map ni score.")
                else:
                    anomaly_map = output
                    score = anomaly_map.max().item()
                    
                scores.append(score)
                labels.append(label)
                
                # 5. Visualisation
                img_np = np.array(img_pil.resize((img_size, img_size)))
                anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
                
                overlay = apply_heatmap(img_np, anomaly_map_np)
                
                # Sauvegarde
                save_dir = out_dir / defect_type
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_dir / img_name), overlay)

    # 6. Bilan
    if num_images == 0:
        print("Aucune image trouvée pour le test.")
        return
        
    avg_latency = (total_time / num_images) * 1000
    fps = num_images / total_time
    
    print("\n" + "="*50)
    print(f" BILAN EVALUATION : {model_name.upper()} - {category.upper()}")
    print("="*50)
    print(f"[Chrono] Temps d'inférence pur : {total_time:.2f} s pour {num_images} images")
    print(f"[Perf] Latence : {avg_latency:.1f} ms / image  |  Débit : {fps:.1f} FPS")
    
    # Calcul FPR avec Zéro Défaut
    print("\n--- Analyse Métriques Zéro Défaut ---")
    best_t = get_best_threshold(scores, labels)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'évaluation pour PaDiM et PatchCore")
    parser.add_argument("--model", type=str, choices=["padim", "patchcore"], required=True, help="Choix du modèle")
    parser.add_argument("--category", type=str, required=True, help="Catégorie (ex: capsule)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin vers le fichier .ckpt")
    parser.add_argument("--output_dir", type=str, default="", help="Dossier de sauvegarde des cartes")
    
    args = parser.parse_args()
    test_model(args.model, args.category, args.checkpoint, args.output_dir)
