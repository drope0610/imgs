import torch
import cv2
import numpy as np
from pathlib import Path
import time
import os
from anomalib.models import EfficientAd

def get_best_threshold(scores, labels):
    best_thresh = min(scores) - 0.01 if scores else 0.0
    min_fp = float('inf')
    thresholds = sorted(list(set(scores)))
    for t in thresholds:
        fp = sum(1 for s, l in zip(scores, labels) if s > t and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s <= t and l == 1)
        if fn == 0:
            if fp < min_fp:
                min_fp = fp
                best_thresh = t
    total_ok = sum(1 for l in labels if l == 0)
    faux_positifs_pct = (min_fp / total_ok) * 100 if total_ok > 0 else 0
    print(f"\n-> Politique Zéro Défaut validée (100% des défauts trouvés).")
    print(f"-> Taux de Faux Positifs (Pièces saines jetées à tort) : {faux_positifs_pct:.1f}% ({min_fp}/{total_ok})")
    return best_thresh

paths_possibles = [
    Path("/media/pedro/Modeles/mvtec_anomaly_detection"),
    Path("/media/pedro2/Modeles/mvtec_anomaly_detection"),
    Path("./mvtec_anomaly_detection")
]
dataset_root = None
for p in paths_possibles:
    if p.exists() and p.is_dir():
        dataset_root = p
        break

# Find the lightning checkpoint dynamically
possible_dirs = [
    Path("/home/pedro/results/efficientad/pill/"),
    Path("/home/pedro/Desktop/Images/imgs/results/efficientad/pill/")
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

test_dir = dataset_root / "pill" / "test"
paths = list(test_dir.glob("**/*.png"))
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
