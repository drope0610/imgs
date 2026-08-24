#!/usr/bin/env python3
"""
Benchmark PillQC - Contrôle Qualité de Pilules (Pill Quality Control)
----------------------------------------------------------------------
Ce script permet d'évaluer la détection d'anomalies de pilules selon les classes de défauts PillQC :
Normal (Sain), Contamination (Dirt), Éclats/Morceaux manquant (Chip), Fissures (Crack).

Usage :
    python benchmarks/pillqc.py --dataset_dir path/to/pillqc --model_path path/to/model.pt
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
    parser = argparse.ArgumentParser(description="Benchmark PillQC (Pill Quality Control)")
    parser.add_argument("--dataset_dir", type=str, default="pillqc_dataset", help="Chemin vers le dataset PillQC")
    parser.add_argument("--model_path", type=str, default="", help="Chemin du modèle (.pt, .onnx, .engine)")
    parser.add_argument("--img_size", type=int, default=256, help="Taille des images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output_csv", type=str, default="benchmarks/results_pillqc.csv", help="Fichier CSV de sortie")
    return parser.parse_args()

class PillQCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, img_size: int = 256):
        self.img_size = img_size
        self.samples = []

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        if not dataset_dir.exists():
            return

        # Parcours des répertoires de classes : normal, dirt, chip, crack, etc.
        for folder in dataset_dir.iterdir():
            if folder.is_dir():
                class_name = folder.name.lower()
                label = 0 if class_name in ["normal", "good", "ok"] else 1
                for img_path in list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")):
                    self.samples.append((img_path, label, folder.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, defect_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label, str(img_path), defect_type

def main():
    args = parse_args()
    print("=" * 60)
    print(" 💊 BENCHMARK PILLQC - CONTRÔLE QUALITÉ DE PILULES")
    print(f" ➜ Dossier Dataset : {args.dataset_dir}")
    print(f" ➜ Modèle          : {args.model_path or 'Modèle de démonstration / Stub'}")
    print(f" ➜ Device          : {args.device}")
    print("=" * 60)

    dataset_dir = Path(args.dataset_dir)
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)

    wrapper = AnomalyModelWrapper(args.model_path, args.device, category="pill")
    wrapper.load()

    dataset = PillQCDataset(dataset_dir, img_size=args.img_size)
    if len(dataset) == 0:
        print(f"[AVERTISSEMENT] Aucune image trouvée dans {dataset_dir}.")
        print("Veuillez indiquer le bon chemin du dataset PillQC via --dataset_dir.")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    total_time = 0.0
    scores, labels = [], []

    with open(args.output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Dataset', 'Classe / Défaut', 'Image', 'Vérité Terrain', 'Score Anomalie', 'Latence (ms)'])

        for tensor, label, img_path, defect_type in dataloader:
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

            writer.writerow(['PillQC', defect_type[0], Path(img_path[0]).name, "NOK" if label.item() == 1 else "OK", f"{score:.4f}", f"{latency:.2f}"])

    avg_latency = total_time / len(dataset) if len(dataset) > 0 else 0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n" + "=" * 60)
    print(" 📊 BILAN DU BENCHMARK PILLQC")
    print(f" ➜ Total Images   : {len(dataset)}")
    print(f" ➜ Latence Moyenne : {avg_latency:.2f} ms")
    print(f" ➜ Débit           : {fps:.1f} FPS")
    print(f" ➜ Résultats       : {args.output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
