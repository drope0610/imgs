import os
import argparse
import shutil
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim, Fastflow, Patchcore

import rich.console
rich.console.Console.clear_live = lambda self: None

if not hasattr(torch, "distributed"):
    import types
    torch.distributed = types.ModuleType("distributed")
if not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False

def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ CUDA est disponible ! GPU détecté : {torch.cuda.get_device_name(0)}")
        return "gpu"
    else:
        print("⚠️ Aucun GPU détecté. L'entraînement se fera sur CPU.")
        return "cpu"

def main():
    device = check_gpu()
    
    dataset_root = Path("datasets/pillQC-main/images")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dossier dataset introuvable : {dataset_root}")

    img_size = 256
    print(f"📦 Configuration du datamodule Folder pour PillQC (Taille: {img_size}x{img_size})...")
    
    # Configuration Folder Datamodule (Anomalib)
    datamodule = Folder(
        name="pillqc",
        root=dataset_root,
        normal_dir="normal",
        abnormal_dir="dirt",
        image_size=(img_size, img_size),
        train_batch_size=4, # Batch size larger to speed up Padim/Fastflow
        eval_batch_size=4,
        num_workers=4,
        task="classification"
    )

    models_to_train = [
        ("padim", Padim()),
        ("fastflow", Fastflow()),
        ("patchcore", Patchcore())
    ]
    
    out_dir = Path("trained_models")
    out_dir.mkdir(exist_ok=True)

    for model_name, model in models_to_train:
        print(f"\n=======================================================")
        print(f"🚀 Lancement de l'entraînement pour : {model_name.upper()}")
        print(f"=======================================================")
        
        # Max epochs depends on the model (Padim is 1 epoch essentially, Fastflow needs ~200-500)
        # However Anomalib handles internal defaults for some, but Engine requires max_epochs
        # Padim and Patchcore extract features, they don't do gradient descent loops over epochs in the same way,
        # but Engine handles it. We can just set a reasonable max_epochs, FastFlow benefits from 250.
        epochs = 250 if model_name == "fastflow" else 1

        engine = Engine(
            accelerator=device,
            devices=1,
            max_epochs=epochs,
            default_root_dir=f"./results/{model_name}/pillqc"
        )

        try:
            engine.fit(model=model, datamodule=datamodule)
            print(f"✅ Entraînement {model_name} terminé !")
            
            # Recherche du meilleur checkpoint
            ckpt_paths = list(Path(f"./results/{model_name}/pillqc").glob("**/*.ckpt"))
            if ckpt_paths:
                ckpt_paths.sort(key=os.path.getmtime, reverse=True)
                best_ckpt = ckpt_paths[0]
                out_path = out_dir / f"pillQC250ep_{model_name}.pt"
                shutil.copy(best_ckpt, out_path)
                print(f"🎉 Modèle sauvegardé avec succès dans : {out_path}")
            else:
                print(f"❌ Erreur : Aucun checkpoint trouvé pour {model_name}.")
                
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement de {model_name}: {e}")
            
    print("\n✅🚀 Campagne Multi-Modèles Terminée ! 🚀✅")

if __name__ == "__main__":
    main()
