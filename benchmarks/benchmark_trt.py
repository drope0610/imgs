import os
import time
import argparse
import numpy as np
import cv2
import glob
from prettytable import PrettyTable
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deploy.trt_engine import TensorRTEngine
from src.config import get_dataset_root, get_results_dir

def load_images(img_paths, img_size=256, max_imgs=100):
    images = []
    for path in img_paths[:max_imgs]:
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img_ready = np.ascontiguousarray(img, dtype=np.float32)
        images.append((path, img_ready))
    return images

def benchmark_engine(engine_path, images, num_warmup=10):
    if not os.path.exists(engine_path):
        return None
        
    engine = TensorRTEngine(engine_path)
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    
    # Warmup
    if images:
        for _ in range(num_warmup):
            engine.infer(images[0][1])
            
    latencies = []
    scores = []
    
    for path, img in images:
        start_time = time.perf_counter()
        outputs = engine.infer(img)
        latency = (time.perf_counter() - start_time) * 1000 # ms
        latencies.append(latency)
        
        # Le score est le max de la carte d'anomalie
        score = float(outputs[0].max())
        scores.append(score)
        
    avg_latency = np.mean(latencies) if latencies else 0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    return {
        "size_mb": size_mb,
        "latency_ms": avg_latency,
        "fps": fps,
        "scores": scores
    }

def get_categories(dataset_root, filter_cat=None):
    if filter_cat:
        return [filter_cat]
    categories = []
    for d in os.listdir(dataset_root):
        cat_path = dataset_root / d
        if cat_path.is_dir() and (cat_path / "test").exists():
            categories.append(d)
    return categories

def main():
    parser = argparse.ArgumentParser(description="Benchmark global TensorRT (FP16 vs INT8)")
    parser.add_argument("--category", type=str, default=None, help="Catégorie spécifique (sinon, teste toutes les catégories)")
    args = parser.parse_args()
    
    dataset_root = get_dataset_root()
    engines_dir = get_results_dir() / "engines"
    
    categories = get_categories(dataset_root, args.category)
    if not categories:
        print("❌ Aucune catégorie valide trouvée.")
        return
        
    print(f"🚀 Début du Benchmark Global (Catégories: {', '.join(categories)})")
    
    table = PrettyTable()
    table.field_names = ["Catégorie", "Modèle", "Taille (Mo)", "Latence (ms)", "Débit (FPS)", "Écart Moyen INT8 vs FP16"]
    
    for cat in categories:
        fp16_engine = engines_dir / f"efficientad_{cat}_fp16.engine"
        int8_engine = engines_dir / f"efficientad_{cat}_int8.engine"
        
        # Charger les images (Mix good/defect)
        good_imgs = glob.glob(str(dataset_root / cat / "test" / "good" / "*.png"))
        defect_imgs = []
        for defect_dir in glob.glob(str(dataset_root / cat / "test" / "*")):
            if "good" not in defect_dir:
                defect_imgs.extend(glob.glob(f"{defect_dir}/*.png"))
                
        test_imgs = load_images(good_imgs[:50] + defect_imgs[:50])
        
        if not test_imgs:
            print(f"⚠️ Aucune image de test trouvée pour {cat}. Skipped.")
            continue
            
        print(f"\n📊 Benchmarking {cat} sur {len(test_imgs)} images...")
        fp16_res = benchmark_engine(str(fp16_engine), test_imgs)
        int8_res = benchmark_engine(str(int8_engine), test_imgs)
        
        # FP16 Row
        if fp16_res:
            table.add_row([
                cat, "FP16", 
                f"{fp16_res['size_mb']:.2f}", 
                f"{fp16_res['latency_ms']:.2f}", 
                f"{fp16_res['fps']:.1f}", 
                "-"
            ])
        else:
            table.add_row([cat, "FP16", "N/A", "N/A", "N/A", "-"])
            
        # INT8 Row
        if int8_res:
            diff_str = "N/A"
            if fp16_res and len(int8_res["scores"]) == len(fp16_res["scores"]):
                diffs = [abs(i - f) for i, f in zip(int8_res["scores"], fp16_res["scores"])]
                avg_diff = np.mean(diffs)
                diff_str = f"± {avg_diff:.4f}"
                
            table.add_row([
                cat, "INT8", 
                f"{int8_res['size_mb']:.2f}", 
                f"{int8_res['latency_ms']:.2f}", 
                f"{int8_res['fps']:.1f}", 
                diff_str
            ])
        else:
            table.add_row([cat, "INT8", "N/A", "N/A", "N/A", "N/A"])
            
    print("\n" + str(table) + "\n")
    print("💡 Astuce : Un faible 'Écart Moyen' signifie que le modèle INT8 a très bien conservé la précision du FP16.")

if __name__ == "__main__":
    main()
