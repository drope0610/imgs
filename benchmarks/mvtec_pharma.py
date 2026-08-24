#!/usr/bin/env python3
"""
Benchmark MVTec AD - Catégories Pharmaceutiques ('pill' & 'capsule')
------------------------------------------------------------------
Ce script permet d'évaluer un modèle personnalisé (PyTorch, ONNX, TensorRT ou Anomalib)
sur les catégories pharmaceutiques du jeu de données MVTec AD.

Usage :
    python benchmarks/mvtec_pharma.py --model_path path/to/model.pt --dataset_dir datasets/mvtec_anomaly_detection
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from model_wrapper import AnomalyModelWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MVTec AD (Pharma: pill & capsule)")
    
    # Auto-detect NVMe or use default
    default_dir = "datasets/mvtec_anomaly_detection"
    possible_dirs = [
        "/media/pedro2/nvme/datasets/mvtec_anomaly_detection",
        "/media/pedro/Modeles/datasets/mvtec_anomaly_detection",
        "/media/pedro2/Modeles/datasets/mvtec_anomaly_detection"
    ]
    for d in possible_dirs:
        if os.path.exists(d):
            default_dir = d
            break
            
    parser.add_argument("--dataset_dir", type=str, default=default_dir, help="Chemin vers la racine de MVTec AD")
    parser.add_argument("--categories", nargs="+", default=["pill", "capsule"], help="Catégories à évaluer")
    parser.add_argument("--model_path", type=str, default="", help="Chemin du modèle (.pt, .pth, .onnx, .engine)")
    parser.add_argument("--img_size", type=int, default=240, help="Taille des images en entrée")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output_csv", type=str, default="benchmarks/results_mvtec_pharma.csv", help="Fichier CSV de sortie")
    return parser.parse_args()

class MVTecPharmaDataset(torch.utils.data.Dataset):
    def __init__(self, category_dir: Path, img_size: int = 240):
        self.img_size = img_size
        self.samples = []
        
        test_dir = category_dir / "test"
        if not test_dir.exists():
            print(f"[Attention] Le dossier {test_dir} n'existe pas.")
            return

        transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.transform = transform

        for defect_type in test_dir.iterdir():
            if defect_type.is_dir():
                label = 0 if defect_type.name == "good" else 1
                for img_path in list(defect_type.glob("*.png")) + list(defect_type.glob("*.jpg")):
                    self.samples.append((img_path, label, defect_type.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, defect_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label, str(img_path), defect_type

def run_dummy_inference(tensor, device):
    """Fonction stub simulant une inférence d'anomalie si aucun modèle n'est chargé."""
    # Renvoie un score d'anomalie aléatoire (0.0 à 1.0)
    return float(np.random.uniform(0.0, 1.0))

def main():
    args = parse_args()
    print("=" * 60)
    print(" 💊 BENCHMARK MVTEC AD - CATÉGORIES PHARMACEUTIQUES")
    print(f" ➜ Catégories : {args.categories}")
    print(f" ➜ Modèle     : {args.model_path or 'Modèle de démonstration / Stub'}")
    print(f" ➜ Device     : {args.device}")
    print("=" * 60)

    dataset_root = Path(args.dataset_dir)
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)

    results_summary = []

    with open(args.output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Catégorie', 'Image', 'Type Defaut', 'Verite Terrain', 'Score Anomalie', 'Latence (ms)'])

        for cat in args.categories:
            cat_dir = dataset_root / cat
            if not cat_dir.exists():
                print(f"\n[Saut] Catégorie '{cat}' introuvable dans {dataset_root}")
                continue

            print(f"\n>>> Évaluation de la catégorie : {cat.upper()} <<<")
            
            # Chargement dynamique du modèle (WinCLIP, PyTorch, TensorRT)
            wrapper = AnomalyModelWrapper(args.model_path, args.device, cat)
            wrapper.load()

            dataset = MVTecPharmaDataset(cat_dir, img_size=args.img_size)
            if len(dataset) == 0:
                print(f"Aucune image de test trouvée pour {cat}.")
                continue

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            total_time = 0.0
            scores = []
            labels = []

            for tensor, label, img_path, defect_type in dataloader:
                tensor = tensor.to(args.device)
                
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()

                # --- Inférence ---
                if args.model_path:
                    # Inférence via le wrapper
                    score = wrapper.predict(tensor)
                else:
                    score = run_dummy_inference(tensor, args.device)

                if args.device == 'cuda':
                    torch.cuda.synchronize()
                latency = (time.time() - t0) * 1000.0
                total_time += latency

                scores.append(score)
                labels.append(label.item())

                writer.writerow([cat, Path(img_path[0]).name, defect_type[0], "NOK" if label.item() == 1 else "OK", f"{score:.4f}", f"{latency:.2f}"])

            avg_latency = total_time / len(dataset)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            
            print(f"   ➜ Images évaluées : {len(dataset)}")
            print(f"   ➜ Latence moyenne  : {avg_latency:.2f} ms/img")
            print(f"   ➜ Débit           : {fps:.1f} FPS")
            results_summary.append((cat, len(dataset), avg_latency, fps))

    print("\n" + "=" * 60)
    print(" 📊 BILAN DU BENCHMARK MVTEC PHARMA")
    for cat, n, lat, fps in results_summary:
        print(f" ➜ {cat.upper():<10} | {n} imgs | Latence: {lat:.2f} ms | FPS: {fps:.1f}")
    print(f" ➜ Résultats détaillés sauvegardés dans : {args.output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
