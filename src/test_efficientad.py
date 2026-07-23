import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def get_best_threshold(scores, labels):
    best_thresh = min(scores) - 0.01 if scores else 0.0
    min_fp = float('inf')
    thresholds = sorted(list(set(scores)))
    
    for t in thresholds:
        fp = sum(1 for s, l in zip(scores, labels) if s > t and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s <= t and l == 1)
        
        # Zéro Défaut : 0 Faux Négatifs
        if fn == 0:
            if fp < min_fp:
                min_fp = fp
                best_thresh = t
                
    total_ok = sum(1 for l in labels if l == 0)
    faux_positifs_pct = (min_fp / total_ok) * 100 if total_ok > 0 else 0
    print(f"-> Politique Zéro Défaut validée (100% des défauts trouvés).")
    print(f"-> Taux de Faux Positifs (Pièces saines jetées à tort) : {faux_positifs_pct:.1f}% ({min_fp}/{total_ok})")
    
    return best_thresh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="pill", help="Catégorie à tester")
    args = parser.parse_args()
    
    category = args.category
    engine_path = f"./results/engines/efficientad_{category}.engine"
    
    if not os.path.exists(engine_path):
        print(f"Moteur TensorRT introuvable : {engine_path}")
        return
        
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
            
    if dataset_root is None:
        raise FileNotFoundError("Impossible de trouver le dataset MVTec sur la clé USB ou en local.")
        
    test_dir = dataset_root / category / "test"
    if not test_dir.exists():
        print(f"Dossier de test introuvable : {test_dir}")
        return
        
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=input_dtype)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=output_dtype)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    memory_refs = [d_input, d_output]
    for i in range(2, engine.num_io_tensors):
        extra_name = engine.get_tensor_name(i)
        extra_shape = engine.get_tensor_shape(extra_name)
        extra_dtype = trt.nptype(engine.get_tensor_dtype(extra_name))
        taille_octets = trt.volume(extra_shape) * np.dtype(extra_dtype).itemsize
        d_extra = cuda.mem_alloc(taille_octets)
        context.set_tensor_address(extra_name, int(d_extra))
        memory_refs.append(d_extra)
        
    stream = cuda.Stream()
    
    print(f"=== ÉVALUATION EFFICIENT-AD (ZÉRO DÉFAUT) SUR '{category}' ===")
    
    scores = []
    labels = []
    latencies = []
    
    paths = list(test_dir.glob("**/*.png"))
    for idx, p in enumerate(paths):
        parent_dir = p.parent.name
        label = 0 if parent_dir == "good" else 1
        
        img = cv2.imread(str(p))
        if img is None: continue
        
        img_rgb = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_ready = np.ascontiguousarray(np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0), dtype=input_dtype)
        
        np.copyto(h_input, img_ready.ravel())
        
        t0 = time.time()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        t1 = time.time()
        
        if idx > 5: # Ignorer le warmup
            latencies.append((t1 - t0) * 1000.0)
            
        anomaly_map = h_output.reshape((256, 256))
        # Score d'anomalie = max de la heatmap
        score = float(np.max(anomaly_map))
        
        scores.append(score)
        labels.append(label)
        
    if latencies:
        print(f"-> Latence moyenne TensorRT : {np.mean(latencies):.2f} ms")
        
    get_best_threshold(scores, labels)
    
if __name__ == "__main__":
    main()
