import argparse
import sys
import os
import cv2
import numpy as np
import glob

sys.path.append("/usr/lib/python3.10/dist-packages")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deploy.trt_engine import TensorRTEngine
from src.config import get_dataset_root, get_results_dir

def main():
    parser = argparse.ArgumentParser(description="Évaluation TensorRT massive avec export TIFF")
    parser.add_argument("--category", type=str, default="capsule", help="Catégorie (ex: capsule)")
    parser.add_argument("--engine", type=str, help="Chemin du .engine (par défaut FP16)")
    parser.add_argument("--output_dir", type=str, help="Dossier de sortie (optionnel)")
    parser.add_argument("--img_size", type=int, default=256, help="Taille de l'image (256 ou 512)")
    args = parser.parse_args()

    from pathlib import Path
    dataset_root = get_dataset_root() / args.category
    
    if args.category == "pillqc":
        test_dir = dataset_root / "images"
    else:
        test_dir = dataset_root / "test"
    
    if args.engine:
        engine_path = args.engine
    else:
        engine_path = str(get_results_dir() / "engines" / f"efficientad_{args.category}_fp16.engine")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(get_results_dir() / "anomaly_maps" / args.category)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Moteur TensorRT introuvable : {engine_path}")

    # 1. Chargement du moteur optimisé (TensorRTEngine gère déjà l'allocation et les streams)
    print(f"⚙️ Chargement du moteur TensorRT : {engine_path}")
    engine = TensorRTEngine(engine_path)
    
    target_h = target_w = args.img_size

    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))
        
    if not image_paths:
        print(f"⚠️ Aucune image de test trouvée dans {test_dir}")
        return

    print(f"🚀 Début de l'évaluation sur {len(image_paths)} images (taille cible: {target_w}x{target_h})...")
    
    import csv
    csv_path = os.path.join(output_dir, "scores.csv")
    csv_data = []

    for img_path in image_paths:
        # Prétraitement (identique à Anomalib)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_w, target_h))
        
        # Pas de normalisation ImageNet manuelle car Anomalib exporte souvent ça dans l'ONNX
        img_norm = img_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_norm, (2, 0, 1))
        img_batched = np.expand_dims(img_input, axis=0)
        
        img_ready = np.ascontiguousarray(img_batched, dtype=engine.inputs[0]['dtype'])

        # Inférence via la classe unifiée
        outputs = engine.infer(img_ready)
        
        # Extraction de l'anomaly map
        output_size = outputs[0].size
        side = int(np.sqrt(output_size))
        anomaly_map = outputs[0].reshape((side, side))
        
        # Le score de l'image est généralement le max de l'anomaly map
        image_score = np.max(anomaly_map)
        
        # Redimensionnement à la taille d'origine (idéal pour le calcul d'AU-ROC par la suite)
        anomaly_map_resized = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]))
        
        # Sauvegarde en 32-bit TIFF Float pour ne perdre AUCUNE PRÉCISION
        category_name = os.path.basename(os.path.dirname(img_path)) # ex: good, hole, scratch
        label = 0 if category_name in ["good", "normal"] else 1
        
        filename = os.path.splitext(os.path.basename(img_path))[0] + '.tiff'
        
        save_path = os.path.join(output_dir, category_name)
        os.makedirs(save_path, exist_ok=True)
        
        # OpenCV supporte l'écriture de float32 dans un TIFF (si TIFF est compilé)
        cv2.imwrite(os.path.join(save_path, filename), anomaly_map_resized)
        
        # Ajouter au CSV
        csv_data.append([filename, category_name, label, image_score])

    # Sauvegarder le CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category", "label", "score"])
        writer.writerows(csv_data)

    print(f"✅ Évaluation terminée ! Toutes les cartes ont été générées dans {output_dir}")
    print(f"📄 Scores image-level sauvegardés dans {csv_path}")

if __name__ == "__main__":
    main()
