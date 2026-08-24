import torch
import cv2
import numpy as np
from pathlib import Path
import time
import os
from anomalib.models import EfficientAd
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_dataset_root
from src.utils.metrics import get_best_threshold

parser = argparse.ArgumentParser(description="Test EfficientAD PyTorch Model")
parser.add_argument("--category", type=str, default="pill", help="Dataset category (e.g. pill, capsule)")
args = parser.parse_args()


dataset_root = get_dataset_root()

from src.config import get_results_dir
possible_dirs = [
    get_results_dir() / "efficientad" / args.category,
    Path.cwd() / f"results/efficientad/{args.category}",
]
ckpt_paths = []
for d in possible_dirs:
    if d.exists():
        ckpt_paths.extend(list(d.glob("**/*.ckpt")))

ckpt_paths = sorted(ckpt_paths, key=os.path.getmtime, reverse=True)
if not ckpt_paths:
    print("Checkpoint introuvable !")
    exit(1)
ckpt_path = ckpt_paths[0]
print(f"Chargement du modèle depuis {ckpt_path}...")
model = EfficientAd.load_from_checkpoint(str(ckpt_path)).cuda().eval()

test_dir = dataset_root / args.category / "test"
paths = list(test_dir.glob("**/*.png")) + list(test_dir.glob("**/*.jpg"))
print(f"Évaluation de {len(paths)} images sur EfficientAD (PyTorch)...")

scores, labels = [], []
latencies = []
for idx, p in enumerate(paths):
    label = 0 if p.parent.name == "good" else 1
    img = cv2.imread(str(p))
    img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
    
    # EfficientAD attends inputs in [0, 1] normalized with ImageNet stats normally, 
    # but anomalib handles its own internal transformations. We just send raw [0,1] or let the model do it.
    # Actually, anomalib models usually expect inputs in [0,1].
    img_tensor = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0).cuda() / 255.0
    
    with torch.no_grad():
        t0 = time.time()
        out = model(img_tensor)
        t1 = time.time()
        
    if idx > 5:
        latencies.append((t1 - t0) * 1000)
        
    if isinstance(out, dict):
        score = out["pred_score"].item() if "pred_score" in out else torch.max(out["anomaly_map"]).item()
    elif hasattr(out, "pred_score"):
        score = out.pred_score.item()
    elif isinstance(out, tuple):
        score = out[0].item() if out[0].numel() == 1 else torch.max(out[1]).item()
    else:
        score = torch.max(out).item()
        
    scores.append(score)
    labels.append(label)

if latencies:
    print(f"-> Latence moyenne PyTorch (sans TensorRT) : {np.mean(latencies):.2f} ms")

get_best_threshold(scores, labels)
