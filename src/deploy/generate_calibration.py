import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

class ImageCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, img_dir, shape=(1, 3, 256, 256), cache_file="calibration.cache"):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.shape = shape
        self.batch_size = shape[0]
        
        # Charger les chemins des images "good" (sans défauts)
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        self.current_idx = 0
        
        # Allouer la mémoire GPU pour le batch
        self.device_input = cuda.mem_alloc(trt.volume(shape) * 4) # 4 bytes pour float32

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx + self.batch_size > len(self.img_paths):
            return None # Fin de la calibration

        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.img_paths[self.current_idx + i]
            # Prétraitement standard Anomalib (à adapter selon vos transforms)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.shape[3], self.shape[2]))
            img = img.astype(np.float32) / 255.0
            # Normalisation ImageNet
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
            batch_imgs.append(img)

        self.current_idx += self.batch_size
        batch_data = np.ascontiguousarray(batch_imgs, dtype=np.float32)
        
        # Copier les données sur le GPU
        cuda.memcpy_htod(self.device_input, batch_data)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# --- Exécution ---
# Laissez TensorRT construire un moteur temporaire pour générer le cache
if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open("model.onnx", "rb") as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = ImageCalibrator("dataset/cable/train/good/", cache_file="efficientad_calib.cache")

    # Le simple fait de construire le moteur déclenche la calibration et crée le fichier .cache
    engine = builder.build_engine(network, config)
    print("Calibration terminée ! Fichier efficientad_calib.cache généré.")
