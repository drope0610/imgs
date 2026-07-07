import torch
import gc
# from anomalib.models import WinClip

def find_max_batch_size(model, input_shape=(3, 256, 256), start_batch=1, max_batch=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    optimal_batch = start_batch
    
    print(f"Recherche du Batch Size optimal pour WinClip sur {device}...")
    
    for b in range(start_batch, max_batch + 1):
        try:
            # Création d'un tenseur dummy pour simuler un batch
            dummy_input = torch.randn(b, *input_shape, device=device)
            
            with torch.no_grad():
                # Forward pass
                _ = model(dummy_input)
                
            print(f"✅ Batch size {b} : OK")
            optimal_batch = b
            
            # Nettoyage préventif
            del dummy_input
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM détecté au batch size {b}.")
                
                # NETTOYAGE CRITIQUE SUR JETSON
                if 'dummy_input' in locals():
                    del dummy_input
                torch.cuda.empty_cache()
                gc.collect() # Force le garbage collector Python
                break
            else:
                # Si c'est une autre erreur RuntimeError, on la remonte
                raise e
                
    print(f"\n🚀 Batch Size maximal recommandé pour la production : {optimal_batch}")
    return optimal_batch

if __name__ == "__main__":
    pass
    # winclip_model = WinClip(...) 
    # max_bs = find_max_batch_size(winclip_model, input_shape=(3, 224, 224))
