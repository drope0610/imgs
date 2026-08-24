import tensorrt as trt
import numpy as np
import ctypes

cudart = ctypes.CDLL('/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12')

cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int

cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
cudart.cudaHostAlloc.restype = ctypes.c_int

cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int

cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
cudart.cudaFreeHost.restype = ctypes.c_int

cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
cudart.cudaMemcpyAsync.restype = ctypes.c_int

cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudart.cudaStreamCreate.restype = ctypes.c_int

cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
cudart.cudaStreamSynchronize.restype = ctypes.c_int

cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
cudart.cudaStreamDestroy.restype = ctypes.c_int

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaHostAllocDefault = 0

def check_cuda_err(err):
    if err != 0:
        raise RuntimeError(f"CUDA Error Code: {err}")

class TensorRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        check_cuda_err(cudart.cudaStreamCreate(ctypes.byref(self.stream)))
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            vol = trt.volume(shape)
            if vol < 0: vol = abs(vol)
            size = vol * np.dtype(dtype).itemsize
            
            host_mem = ctypes.c_void_p()
            check_cuda_err(cudart.cudaHostAlloc(ctypes.byref(host_mem), size, cudaHostAllocDefault))
            
            buffer = (ctypes.c_char * size).from_address(host_mem.value)
            host_array = np.ndarray(buffer=buffer, dtype=dtype, shape=shape)
            
            device_mem = ctypes.c_void_p()
            check_cuda_err(cudart.cudaMalloc(ctypes.byref(device_mem), size))
            
            self.bindings.append(device_mem.value)
            self.context.set_tensor_address(name, device_mem.value)
            
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            binding = {
                'name': name,
                'host_mem': host_mem.value,
                'host_array': host_array,
                'device_mem': device_mem.value,
                'shape': shape,
                'dtype': dtype,
                'size': size,
                'is_input': is_input
            }
            if is_input: self.inputs.append(binding)
            else: self.outputs.append(binding)

    def infer(self, input_data):
        if not isinstance(input_data, list): input_data = [input_data]
            
        for i, inp in enumerate(self.inputs):
            data = input_data[i]
            if data.dtype != inp['dtype']: data = data.astype(inp['dtype'])
            np.copyto(inp['host_array'], data.reshape(inp['shape']))
            check_cuda_err(cudart.cudaMemcpyAsync(ctypes.c_void_p(inp['device_mem']), ctypes.c_void_p(inp['host_mem']), inp['size'], cudaMemcpyHostToDevice, self.stream))
            
        self.context.execute_async_v3(stream_handle=self.stream.value)
        
        for out in self.outputs:
            check_cuda_err(cudart.cudaMemcpyAsync(ctypes.c_void_p(out['host_mem']), ctypes.c_void_p(out['device_mem']), out['size'], cudaMemcpyDeviceToHost, self.stream))
            
        check_cuda_err(cudart.cudaStreamSynchronize(self.stream))
        return [np.copy(out['host_array']) for out in self.outputs]

    def __del__(self):
        if hasattr(self, 'inputs'):
            for out in self.outputs + self.inputs:
                cudart.cudaFreeHost(ctypes.c_void_p(out['host_mem']))
                cudart.cudaFree(ctypes.c_void_p(out['device_mem']))
        if hasattr(self, 'stream') and hasattr(self.stream, 'value') and self.stream.value is not None:
            cudart.cudaStreamDestroy(self.stream)
