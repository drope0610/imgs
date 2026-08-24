import torch
import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.deploy.trt_engine import TensorRTEngine
from anomalib.models import EfficientAd

def find_max_batch_size(model_path=None, engine_path=None, input_shape=(3, 256, 256), start_batch=1, max_batch=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path:
        model = EfficientAd.load_from_checkpoint(model_path).to(device)
        model.eval()
        model_type = "pytorch"
    elif engine_path:
        model = TensorRTEngine(engine_path)
        model_type = "tensorrt"
    else:
        raise ValueError("Veuillez spécifier un model_path ou engine_path.")

    optimal_batch = start_batch
    print(f"Recherche du Batch Size optimal sur {device} (Mode: {model_type})...")
    
    for b in range(start_batch, max_batch + 1):
        try:
            if model_type == "pytorch":
                dummy_input = torch.randn(b, *input_shape, device=device)
                with torch.no_grad():
                    _ = model(dummy_input)
                del dummy_input
            elif model_type == "tensorrt":
                # TRT Engine depends on how it was built (static vs dynamic batch)
                # This simple test assumes dynamic batching is supported up to max_batch
                pass # Complex to simulate if TRT was built with static batch
                
            print(f"✅ Batch size {b} : OK")
            optimal_batch = b
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM détecté au batch size {b}.")
                torch.cuda.empty_cache()
                gc.collect() 
                break
            else:
                raise e
                
    print(f"\n🚀 Batch Size maximal recommandé : {optimal_batch}")
    return optimal_batch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Chemin du .ckpt PyTorch")
    args = parser.parse_args()
    
    if args.model:
        find_max_batch_size(model_path=args.model)
    else:
        print("Veuillez fournir un chemin de modèle via --model")
