import argparse
import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.deploy.trt_engine import TensorRTEngine
from src.config import get_dataset_root, get_results_dir

def main():
    parser = argparse.ArgumentParser(description="Inférence unitaire TensorRT pour Anomalib")
    parser.add_argument("--category", type=str, default="capsule", help="Catégorie (ex: capsule)")
    parser.add_argument("--image", type=str, help="Chemin vers l'image de test. Si vide, cherche la première image.")
    parser.add_argument("--engine", type=str, help="Chemin vers le fichier .engine. Si vide, cherche le fp16.")
    parser.add_argument("--output", type=str, default="resultat_anomalie.png", help="Fichier de sortie")
    args = parser.parse_args()

    # Résolution des chemins dynamiques
    if args.engine:
        engine_path = args.engine
    else:
        engine_path = str(get_results_dir() / "engines" / f"efficientad_{args.category}_fp16.engine")

    if args.image:
        image_path = args.image
    else:
        dataset_root = get_dataset_root() / args.category / "test"
        # Chercher un défaut quelconque (hors dossier "good")
        import glob
        defect_dirs = [d for d in glob.glob(f"{dataset_root}/*") if "good" not in d]
        if defect_dirs:
            imgs = glob.glob(f"{defect_dirs[0]}/*.png")
            if imgs:
                image_path = imgs[0]
            else:
                raise FileNotFoundError(f"Aucune image trouvée dans {defect_dirs[0]}")
        else:
            raise FileNotFoundError(f"Aucun dossier de défaut trouvé dans {dataset_root}")

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Moteur TensorRT introuvable : {engine_path}")

    # 1. Chargement du moteur
    print(f"⚙️ Chargement du moteur TensorRT : {engine_path}")
    engine = TensorRTEngine(engine_path)

    # 2. Préparation de l'image
    print(f"📸 Préparation de l'image : {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image illisible : {image_path}")

    original_img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Normalisation
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)

    # Conversion stricte dans le format demandé par le moteur (dynamique)
    img_ready = np.ascontiguousarray(img_batched, dtype=engine.inputs[0]['dtype'])
    
    # 3. Inférence asynchrone
    print("⚡ Inférence sur la Jetson Orin...")
    outputs = engine.infer(img_ready)
    
    # EfficientAD donne en sortie la carte d'anomalie
    anomaly_map = outputs[0].reshape((256, 256))
    
    # 4. Post-traitement et Carte de chaleur
    print("🎨 Génération de la carte de chaleur...")
    
    map_min, map_max = anomaly_map.min(), anomaly_map.max()
    if map_max - map_min > 0:
        anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
        
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    cv2.imwrite(args.output, overlay)
    print(f"✅ Terminé ! Le résultat visuel est sauvegardé sous '{args.output}'")
    print(f"Score global d'anomalie : {float(outputs[0].max()):.4f}")

if __name__ == "__main__":
    main()
