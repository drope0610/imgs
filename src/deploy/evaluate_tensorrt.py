import argparse
import sys
import os
import cv2
import numpy as np
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.deploy.trt_engine import TensorRTEngine
from src.config import get_dataset_root, get_results_dir

def main():
    parser = argparse.ArgumentParser(description="Évaluation TensorRT massive avec export TIFF")
    parser.add_argument("--category", type=str, default="capsule", help="Catégorie (ex: capsule)")
    parser.add_argument("--engine", type=str, help="Chemin du .engine (par défaut FP16)")
    parser.add_argument("--output_dir", type=str, help="Dossier de sortie (optionnel)")
    args = parser.parse_args()

    dataset_root = get_dataset_root() / args.category
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

    image_paths = glob.glob(os.path.join(test_dir, "**", "*.png"), recursive=True)
    if not image_paths:
        print(f"⚠️ Aucune image de test trouvée dans {test_dir}")
        return

    print(f"🚀 Début de l'évaluation sur {len(image_paths)} images...")

    for img_path in image_paths:
        # Prétraitement (identique à Anomalib)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        
        # Normalisation ImageNet (Standard Anomalib EfficientAD)
        img_norm = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_input = np.transpose(img_norm, (2, 0, 1))
        img_batched = np.expand_dims(img_input, axis=0)
        
        img_ready = np.ascontiguousarray(img_batched, dtype=engine.inputs[0]['dtype'])

        # Inférence via la classe unifiée
        outputs = engine.infer(img_ready)
        
        # Extraction de l'anomaly map
        anomaly_map = outputs[0].reshape((256, 256))
        
        # Redimensionnement à la taille d'origine (idéal pour le calcul d'AU-ROC par la suite)
        anomaly_map_resized = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]))
        
        # Sauvegarde en 32-bit TIFF Float pour ne perdre AUCUNE PRÉCISION
        category_name = os.path.basename(os.path.dirname(img_path)) # ex: good, hole, scratch
        filename = os.path.basename(img_path).replace('.png', '.tiff')
        
        save_path = os.path.join(output_dir, category_name)
        os.makedirs(save_path, exist_ok=True)
        
        # OpenCV supporte l'écriture de float32 dans un TIFF (si TIFF est compilé)
        # Sinon, pour la compatibilité, on sauvegarde la matrice raw via numpy ou on utilise cv2.imwrite
        # Les Tiffs float32 sont standard en analyse d'image
        cv2.imwrite(os.path.join(save_path, filename), anomaly_map_resized)

    print(f"✅ Évaluation terminée ! Toutes les cartes ont été générées dans {output_dir}")

if __name__ == "__main__":
    main()
