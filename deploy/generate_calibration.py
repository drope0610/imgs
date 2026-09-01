import tensorrt as trt
import numpy as np
import cv2
import os
import ctypes

cudart = ctypes.CDLL('/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12')
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype = ctypes.c_int

cudaMemcpyHostToDevice = 1

def check_cuda_err(err):
    if err != 0:
        raise RuntimeError(f"CUDA Error Code: {err}")

try:
    BaseCalibrator = trt.IInt8MinMaxCalibrator
except AttributeError:
    BaseCalibrator = object

class ImageCalibrator(BaseCalibrator):
    def __init__(self, img_dir, shape=(1, 3, 256, 256), cache_file="calibration.cache"):
        if BaseCalibrator is object:
            raise NotImplementedError("IInt8MinMaxCalibrator is missing in this TensorRT version.")
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.shape = shape
        self.batch_size = shape[0]
        
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        self.current_idx = 0
        
        size = trt.volume(shape) * 4
        self.device_input = ctypes.c_void_p()
        check_cuda_err(cudart.cudaMalloc(ctypes.byref(self.device_input), size))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx + self.batch_size > len(self.img_paths):
            return None

        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.img_paths[self.current_idx + i]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.shape[3], self.shape[2]))
            img = img.astype(np.float32) / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = np.transpose(img, (2, 0, 1))
            batch_imgs.append(img)

        self.current_idx += self.batch_size
        batch_data = np.ascontiguousarray(batch_imgs, dtype=np.float32)
        
        check_cuda_err(cudart.cudaMemcpy(self.device_input, batch_data.ctypes.data, batch_data.nbytes, cudaMemcpyHostToDevice))
        return [self.device_input.value]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def get_calibrator(self):
        return self

    def __del__(self):
        if hasattr(self, 'device_input') and self.device_input.value is not None:
            cudart.cudaFree(self.device_input)
