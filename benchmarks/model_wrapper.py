import torch
import numpy as np
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deploy.trt_engine import TensorRTEngine

class AnomalyModelWrapper:
    def __init__(self, model_arg, device="cuda", category="object"):
        self.model_arg = model_arg.lower() if model_arg and model_arg.lower() == "winclip" else model_arg
        self.device = device
        self.category = category
        self.model_type = self._determine_type()
        self.model = None
        
        # TensorRT specific
        self.trt_engine = None

    def _determine_type(self):
        if not self.model_arg:
            return "stub"
        if self.model_arg == "winclip":
            return "winclip"
        elif self.model_arg.endswith(".engine"):
            return "tensorrt"
        else:
            return "pytorch"

    def load(self):
        print(f"[INFO] Initialisation du modèle en mode : {self.model_type.upper()} pour la catégorie '{self.category}'")
        
        if self.model_type == "stub":
            print("[INFO] Aucun modèle fourni. Utilisation d'un stub générant des scores aléatoires.")
            return

        if self.model_type == "winclip":
            from anomalib.models import WinClip
            try:
                self.model = WinClip(class_name=self.category, scales=tuple())
            except Exception:
                self.model = WinClip(class_name=self.category)
            self.model.to(self.device)
            self.model.setup("predict")
            self.model.eval()
            if self.device == "cuda":
                self.model = self.model.half()
                
        elif self.model_type == "pytorch":
            if not os.path.exists(self.model_arg):
                print(f"[ERREUR] Le fichier {self.model_arg} est introuvable.")
                self.model_type = "stub"
                return
            try:
                from anomalib.models import EfficientAd
                self.model = EfficientAd.load_from_checkpoint(self.model_arg).to(self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
            except Exception as e:
                print(f"[AVERTISSEMENT] Chargement du modèle PyTorch échoué : {e}. Fallback en mode stub.")
                self.model_type = "stub"
                
        elif self.model_type == "tensorrt":
            if not os.path.exists(self.model_arg):
                print(f"[ERREUR] Le fichier {self.model_arg} est introuvable.")
                self.model_type = "stub"
                return
            
            self.trt_engine = TensorRTEngine(self.model_arg)

    def predict(self, tensor):
        if self.model_type == "stub":
            return float(np.random.uniform(0.0, 1.0))
            
        if self.model_type == "winclip":
            if self.device == "cuda":
                tensor = tensor.half()
            with torch.no_grad():
                out = self.model(tensor)
            return self._extract_score(out)
            
        elif self.model_type == "pytorch":
            with torch.no_grad():
                out = self.model(tensor)
            return self._extract_score(out)
            
        elif self.model_type == "tensorrt":
            img_np = tensor.cpu().numpy()
            
            outputs = self.trt_engine.infer(img_np)
            return float(outputs[0].max())

    def _extract_score(self, out):
        if isinstance(out, tuple):
            if len(out) > 0:
                if out[0].numel() == 1:
                    return float(out[0].item())
                else:
                    return float(out[0].max().item())
        elif hasattr(out, "pred_score"):
            return float(out.pred_score.item())
        elif isinstance(out, dict):
            if "pred_score" in out:
                return float(out["pred_score"].item())
            elif "anomaly_map" in out:
                return float(out["anomaly_map"].max().item())
        elif isinstance(out, torch.Tensor):
            if out.numel() == 1:
                return float(out.item())
            return float(out.mean().item()) # Using mean() for fallback as in original scripts
        return 0.0
