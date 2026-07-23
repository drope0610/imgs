import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

onnx_path = "/home/pedro/Desktop/Images/imgs/results/efficientad/pill/EfficientAd/MVTec/pill/v6/weights/onnx/model.onnx"
with open(onnx_path, "rb") as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
            
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
engine = builder.build_serialized_network(network, config)

os.makedirs("/home/pedro/Desktop/Images/imgs/results/engines", exist_ok=True)
with open("/home/pedro/Desktop/Images/imgs/results/engines/efficientad_pill.engine", "wb") as f:
    f.write(engine)
print("Engine compiled successfully.")
