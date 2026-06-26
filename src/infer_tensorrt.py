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

def preprocess_image(image_path):
    # Lecture avec OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    # Sauvegarde de l'image originale pour l'affichage final
    original_img = cv2.resize(img, (256, 256))
    
    # Préparation pour le réseau : BGR -> RGB, Resize 256x256, Normalisation 0-1
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Format PyTorch/TensorRT : (Batch, Channels, Height, Width)
    img = np.transpose(img, (2, 0, 1))
    
    # CORRECTION : expand_dims au lieu de expand_dict !
    img = np.expand_dims(img, axis=0)
    
    # Important : Convertir en Float16 si le moteur a été compilé avec --fp16 !
    return np.ascontiguousarray(img, dtype=np.float16), original_img

def main():
    # 1. Chargement du moteur et création du contexte d'exécution
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # 2. Allocation de la mémoire sur le GPU (VRAM)
    input_binding_idx = 0
    output_binding_idx = 1
    
    input_shape = engine.get_tensor_shape(engine.get_tensor_name(input_binding_idx))
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(output_binding_idx))
    
    # Réservation de la mémoire RAM (Host) et VRAM (Device)
    d_input = cuda.mem_alloc(trt.volume(input_shape) * np.dtype(np.float16).itemsize)
    
    # Attention, la sortie est souvent en Float32 même si l'entrée est en FP16
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    bindings = [int(d_input), int(d_output)]

    # 3. Préparation de l'image
    print("📸 Préparation de l'image...")
    img_input, original_img = preprocess_image(IMAGE_PATH)

    # 4. L'Inférence !
    print("⚡ Inférence sur la Jetson Orin...")
    stream = cuda.Stream()
    
    # Transfert RAM -> VRAM (Copie de l'image sur le GPU)
    cuda.memcpy_htod_async(d_input, img_input, stream)
    
    # Exécution du réseau de neurones
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfert VRAM -> RAM (Récupération de la carte d'anomalie)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # 5. Post-traitement et Affichage (Heatmap)
    print("🎨 Génération de la carte de chaleur...")
    
    # La sortie est un tableau 1D, on le reformate en image 2D (256x256)
    anomaly_map = h_output.reshape((256, 256))
    
    # Normalisation de la carte entre 0 et 255 pour OpenCV
    map_min, map_max = anomaly_map.min(), anomaly_map.max()
    if map_max - map_min > 0:
        anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    
    # Application d'un filtre de couleur (Bleu = OK, Rouge = Défaut)
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    
    # Fusion de la carte de chaleur avec l'image originale (50% transparence)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    # Sauvegarde du résultat
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"✅ Terminé ! Le résultat visuel est sauvegardé sous '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()
