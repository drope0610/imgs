import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# Paramètres globaux
ENGINE_PATH = "./results/efficientad/bottle/efficientad_bottle.engine"
IMAGE_PATH = "./mvtec_anomaly_detection/bottle/test/broken_large/000.png" # Test sur une bouteille cassée
OUTPUT_PATH = "resultat_anomalie.png"

# Initialisation du logger TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    print(f"⚙️ Chargement du moteur TensorRT : {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    # 1. Chargement du moteur et contexte
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # 2. Récupération DYNAMIQUE des infos d'entrée/sortie
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    # trt.nptype permet de traduire le type TensorRT en type NumPy (souvent float32)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

    print(f"📊 Info Entrée : Shape {input_shape} | Type {input_dtype}")
    print(f"📊 Info Sortie  : Shape {output_shape} | Type {output_dtype}")

    # 3. Allocation de la mémoire sécurisée (Pagelocked)
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=input_dtype)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=output_dtype)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # 4. Préparation de l'image
    print("📸 Préparation de l'image...")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH}")

    original_img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Normalisation
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_batched = np.expand_dims(img_transposed, axis=0)

    # Conversion stricte dans le format demandé par le moteur
    img_ready = np.ascontiguousarray(img_batched, dtype=input_dtype)
    
    # Copie des pixels de l'image dans la mémoire Pagelocked
    np.copyto(h_input, img_ready.ravel())

    # 5. Inférence !
    print("⚡ Inférence sur la Jetson Orin...")
    stream = cuda.Stream()
    
    # Copie Host -> Device depuis la mémoire pagelocked (h_input)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # Exécution
    context.execute_async_v3(stream_handle=stream.handle)
    
    # Copie Device -> Host vers la mémoire pagelocked (h_output)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # 6. Post-traitement et Carte de chaleur
    print("🎨 Génération de la carte de chaleur...")
    anomaly_map = h_output.reshape((256, 256))
    
    # Éviter la division par zéro lors de la normalisation
    map_min, map_max = anomaly_map.min(), anomaly_map.max()
    if map_max - map_min > 0:
        anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
        
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"✅ Terminé ! Le résultat visuel est sauvegardé sous '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()
