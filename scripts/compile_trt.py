import torch
from anomalib.models import EfficientAd
import tensorrt as trt
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deploy.generate_calibration import ImageCalibrator
from src.config import get_dataset_root

def export_onnx(model_path, onnx_path, img_size=256):
    print(f"[INFO] Chargement du modèle PyTorch: {model_path}")
    model = EfficientAd.load_from_checkpoint(model_path)
    model.eval()
    model.to('cuda')
    
    dummy_input = torch.randn(1, 3, img_size, img_size, device='cuda')
    
    print(f"[INFO] Exportation vers ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("[INFO] Export ONNX terminé avec succès.")

def build_engine(onnx_path, engine_path, use_int8=False, dataset_dir=None):
    print(f"[INFO] Construction du moteur TensorRT: {engine_path} (INT8: {use_int8})")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    
    # FP16 is baseline
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    if use_int8 and builder.platform_has_fast_int8:
        print("[INFO] Activation du mode INT8 avec Calibration.")
        config.set_flag(trt.BuilderFlag.INT8)
        img_dir = os.path.join(dataset_dir, "train", "good")
        calib_cache = engine_path.replace(".engine", "_calib.cache")
        config.int8_calibrator = ImageCalibrator(img_dir, cache_file=calib_cache)
        
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("[ERREUR] Impossible de parser le fichier ONNX:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
            
    print("[INFO] Début de la compilation TensorRT (Cela peut prendre plusieurs minutes)...")
    engineString = builder.build_serialized_network(network, config)
    
    if engineString is None:
        print("[ERREUR] La compilation a échoué.")
        return False
        
    with open(engine_path, "wb") as f:
        f.write(engineString)
    print(f"[INFO] Moteur TensorRT sauvegardé avec succès : {engine_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Chemin du modèle .pt PyTorch ou .onnx")
    parser.add_argument("--output", required=True, help="Chemin du moteur .engine de sortie")
    parser.add_argument("--int8", action="store_true", help="Activer la compilation INT8")
    parser.add_argument("--category", required=True, help="Catégorie (ex: capsule) pour récupérer la calibration")
    args = parser.parse_args()
    
    if args.model.endswith(".pt") or args.model.endswith(".ckpt"):
        onnx_file = args.model.replace(".pt", ".onnx").replace(".ckpt", ".onnx")
        # 1. Export to ONNX
        export_onnx(args.model, onnx_file)
    else:
        onnx_file = args.model
    
    # Resolve dataset dir for calibration
    dataset_dir = os.path.join(str(get_dataset_root()), args.category)
    
    # 2. Build TRT Engine
    build_engine(onnx_file, args.output, use_int8=args.int8, dataset_dir=dataset_dir)
    
    # 3. Clean ONNX
    if os.path.exists(onnx_file) and args.model != onnx_file:
        os.remove(onnx_file)
    print("[INFO] Processus de compilation TensorRT terminé !")
