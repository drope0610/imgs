import os
import glob
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from pathlib import Path
import time

def find_engines(base_dir):
    return glob.glob(os.path.join(base_dir, '**', '*.engine'), recursive=True)

def evaluate_zero_defect(scores_csv):
    df = pd.read_csv(scores_csv).dropna(subset=['score', 'label'])
    
    y_true = df['label'].values
    y_scores = df['score'].values
    
    if len(np.unique(y_true)) < 2:
        return None, None, None
        
    auroc = roc_auc_score(y_true, y_scores)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Trouver l'indice où le recall est à 1.0
    valid_idx = np.where(recalls >= 0.9999)[0]
    if len(valid_idx) == 0:
        return auroc, 0.0, 0.0
        
    # Parmi ces indices, on veut la meilleure précision possible
    best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    
    zero_defect_precision = precisions[best_idx]
    # Handle the fact that thresholds has len = len(precisions) - 1
    zero_defect_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    return auroc, zero_defect_precision, zero_defect_threshold

def main():
    print("==========================================================")
    print("🚀 PIPELINE DE BENCHMARK GLOBAL (RECHERCHE ZERO DEFAUT) 🚀")
    print("==========================================================")
    
    results_dir = Path.home() / "Desktop" / "Images" / "imgs" / "results"
    if not results_dir.exists():
        results_dir = Path.home() / "results"
        
    evals_dir = results_dir / "evaluations"
    evals_dir.mkdir(parents=True, exist_ok=True)
    
    engines = find_engines(str(results_dir))
    print(f"Trouvé {len(engines)} modèles TensorRT (.engine) à évaluer.\n")
    
    results_table = []
    
    for engine_path in engines:
        # Extraire le nom du modèle depuis le chemin
        # Exemple: .../results/efficientad/pill/efficientad_pill_256_500ep/EfficientAd/.../model.engine
        parts = Path(engine_path).parts
        
        # Heuristique pour trouver le nom court du modèle
        model_name = "unknown"
        category = "pill"
        
        if "efficientad" in engine_path.lower():
            # Chercher le dossier qui contient "ep"
            for p in parts:
                if "ep" in p and ("efficientad" in p or "pill" in p):
                    model_name = p
                    break
        elif "padim" in engine_path.lower():
            model_name = "padim_" + parts[parts.index("padim") + 1]
        elif "patchcore" in engine_path.lower():
            model_name = "patchcore_" + parts[parts.index("patchcore") + 1]
            
        if "pillqc" in engine_path.lower():
            category = "pillqc"
            
        print(f"--- Évaluation du modèle : {model_name} ({category}) ---")
        
        output_dir = evals_dir / model_name
        scores_csv = output_dir / "scores.csv"
        
        img_size = "512" if "512" in model_name else "256"
        
        # Lancer l'évaluation si elle n'a pas déjà été faite
        if not scores_csv.exists():
            start_time = time.time()
            cmd = [
                "python3", "deploy/evaluate_tensorrt.py",
                "--engine", engine_path,
                "--category", category,
                "--output_dir", str(output_dir),
                "--img_size", img_size
            ]
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:" + env.get("LD_LIBRARY_PATH", "")
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError:
                print(f"❌ Erreur lors de l'évaluation de {model_name}")
                continue
            inf_time = time.time() - start_time
            print(f"⏱️ Temps d'inférence total : {inf_time:.2f} s")
        else:
            print(f"✅ Évaluation déjà existante pour {model_name}. Lecture des scores...")
            
        # Analyser les résultats
        metrics = evaluate_zero_defect(str(scores_csv))
        if metrics[0] is not None:
            auroc, zd_precision, zd_thresh = metrics
            print(f"📊 AUROC Image : {auroc:.4f} | Précision Zéro Défaut : {zd_precision:.4f} (Seuil: {zd_thresh:.4f})\n")
            
            results_table.append({
                "Modèle": model_name,
                "Catégorie": category,
                "AUROC": f"{auroc*100:.2f}%",
                "Précision_Zero_Defaut": f"{zd_precision*100:.2f}%",
                "Seuil_Optimal": f"{zd_thresh:.4f}"
            })
        else:
            print("⚠️ Impossible de calculer les métriques (pas assez de classes).\n")
            
    # Sauvegarder le classement
    if results_table:
        df_results = pd.DataFrame(results_table)
        df_results = df_results.sort_values(by=["Catégorie", "Précision_Zero_Defaut", "AUROC"], ascending=[True, False, False])
        
        report_path = evals_dir / "CLASSEMENT_FINAL.md"
        with open(report_path, "w") as f:
            f.write("# 🏆 Classement Final des Modèles (Zéro Défaut)\n\n")
            f.write(df_results.to_markdown(index=False))
            
        print("==========================================================")
        print(f"✅ BENCHMARK TERMINÉ ! Classement sauvegardé dans : {report_path}")
        print("==========================================================")

if __name__ == "__main__":
    main()
