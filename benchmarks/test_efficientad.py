import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_dataset_root
from src.utils.metrics import get_best_threshold

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="pill", help="Catégorie à tester")
    parser.add_argument("--engine", type=str, default=None, help="Chemin vers le fichier .engine")
    args = parser.parse_args()
    
    category = args.category
    if args.engine:
        engine_path = args.engine
    else:
        engine_path = str(get_results_dir() / "engines" / f"efficientad_{category}_fp16.engine")
    
    if not os.path.exists(engine_path):
        print(f"Moteur TensorRT introuvable : {engine_path}")
        return
        
    dataset_root = get_dataset_root()
            
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
