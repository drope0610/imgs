import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import glob

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Mémoire pagelocked (host) pour des transferts CPU <-> GPU très rapides sur Jetson
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem, 'shape': engine.get_binding_shape(binding)})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'shape': engine.get_binding_shape(binding)})
            
    return inputs, outputs, bindings, stream

def infer(context, bindings, inputs, outputs, stream):
    # Transfert asynchrone Host -> Device
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    
    # Exécution de l'inférence TensorRT
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfert asynchrone Device -> Host
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        
    stream.synchronize()
    return [out['host'] for out in outputs]

if __name__ == "__main__":
    # --- Initialisation ---
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open("results/engines/efficientad_cable.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    test_dir = "dataset/cable/test/"
    output_dir = "results/anomaly_maps/cable/"
    os.makedirs(output_dir, exist_ok=True)

    # Boucle sur toutes les images (good et défauts)
    image_paths = glob.glob(os.path.join(test_dir, "**", "*.png"), recursive=True)

    for img_path in image_paths:
        # 1. Prétraitement
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_norm = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_input = np.transpose(img_norm, (2, 0, 1)).ravel()

        # 2. Copie dans la mémoire pagelocked
        np.copyto(inputs[0]['host'], img_input)

        # 3. Inférence
        trt_outputs = infer(context, bindings, inputs, outputs, stream)
        
        # 4. Post-traitement de l'Anomaly Map (adapter selon la sortie d'EfficientAd)
        anomaly_map = trt_outputs[0].reshape((256, 256))
        
        # Redimensionnement à la taille de l'image d'origine pour le calcul AU-ROC (Anomalib standard)
        anomaly_map_resized = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]))
        
        # Normalisation visuelle (optionnelle) et sauvegarde en TIFF 32-bit pour garder la précision
        category_name = os.path.basename(os.path.dirname(img_path)) # ex: "good", "hole", "scratch"
        filename = os.path.basename(img_path).replace('.png', '.tiff')
        
        save_path = os.path.join(output_dir, category_name)
        os.makedirs(save_path, exist_ok=True)
        
        cv2.imwrite(os.path.join(save_path, filename), anomaly_map_resized)

    print(f"Évaluation terminée sur {len(image_paths)} images.")
