#!/usr/bin/env python3
"""
Benchmark VisA - Catégorie 'capsules' (Visual Anomaly Dataset)
--------------------------------------------------------------
Ce script évalue la détection d'anomalies visuelles complexes sur la catégorie
'capsules' du jeu de données VisA (Amazon).

Usage :
    python benchmarks/visa_capsules.py --dataset_dir path/to/visa --model_path path/to/model.pt
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
    parser = argparse.ArgumentParser(description="Benchmark VisA (Category: capsules)")
    parser.add_argument("--dataset_dir", type=str, default="visa_dataset", help="Chemin vers la racine du dataset VisA")
    parser.add_argument("--category", type=str, default="capsules", help="Catégorie VisA à évaluer")
    parser.add_argument("--model_path", type=str, default="", help="Chemin du modèle (.pt, .onnx, .engine)")
    parser.add_argument("--img_size", type=int, default=240, help="Taille des images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output_csv", type=str, default="benchmarks/results_visa_capsules.csv", help="Fichier CSV de sortie")
    return parser.parse_args()

class VisACapsulesDataset(torch.utils.data.Dataset):
    def __init__(self, category_dir: Path, img_size: int = 240):
        self.img_size = img_size
        self.samples = []
        
        # Structure VisA typique : category/Data/Images/Normal et category/Data/Images/Anomaly
        test_dir = category_dir / "Data" / "Images"
        if not test_dir.exists():
            # Alternative structure
            test_dir = category_dir

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # Scan pour trouver les images
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = Path(root) / file
                    label = 0 if "good" in str(img_path).lower() or "normal" in str(img_path).lower() else 1
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label, str(img_path)

def main():
    args = parse_args()
    print("=" * 60)
    print(" 💊 BENCHMARK VISA (Visual Anomaly) - CATÉGORIE 'CAPSULES'")
    print(f" ➜ Dossier Dataset : {args.dataset_dir}")
    print(f" ➜ Modèle          : {args.model_path or 'Modèle de démonstration / Stub'}")
    print(f" ➜ Device          : {args.device}")
    print("=" * 60)

    category_dir = Path(args.dataset_dir) / args.category
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)

    wrapper = AnomalyModelWrapper(args.model_path, args.device, args.category)
    wrapper.load()

    dataset = VisACapsulesDataset(category_dir, img_size=args.img_size)
    if len(dataset) == 0:
        print(f"[AVERTISSEMENT] Aucune image trouvée sous {category_dir}.")
        print("Veuillez télécharger le dataset VisA et indiquer le bon chemin via --dataset_dir.")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    total_time = 0.0
    scores, labels = [], []

    with open(args.output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Dataset', 'Catégorie', 'Image', 'Vérité Terrain', 'Score Anomalie', 'Latence (ms)'])

        for tensor, label, img_path in dataloader:
            tensor = tensor.to(args.device)
            
            if args.device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()

            if args.model_path:
                score = wrapper.predict(tensor)
            else:
                score = float(np.random.uniform(0.0, 1.0))

            if args.device == 'cuda':
                torch.cuda.synchronize()
            latency = (time.time() - t0) * 1000.0
            total_time += latency

            scores.append(score)
            labels.append(label.item())

            writer.writerow(['VisA', args.category, Path(img_path[0]).name, "NOK" if label.item() == 1 else "OK", f"{score:.4f}", f"{latency:.2f}"])

    avg_latency = total_time / len(dataset) if len(dataset) > 0 else 0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n" + "=" * 60)
    print(" 📊 BILAN DU BENCHMARK VISA (CAPSULES)")
    print(f" ➜ Total Images   : {len(dataset)}")
    print(f" ➜ Latence Moyenne : {avg_latency:.2f} ms")
    print(f" ➜ Débit           : {fps:.1f} FPS")
    print(f" ➜ Résultats       : {args.output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
