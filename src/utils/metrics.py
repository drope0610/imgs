def get_best_threshold(scores, labels, target_fpr=None):
    """
    Calcule le meilleur seuil pour la politique "Zéro Défaut" (Rappel de 100% sur les défauts).
    Si un target_fpr est fourni (optionnel), cette fonction pourrait aussi chercher d'autres compromis.
    
    Args:
        scores (list): Liste des scores d'anomalie.
        labels (list): Liste des labels (0 = normal, 1 = anomalie).
        
    Returns:
        float: Le meilleur seuil.
    """
    best_thresh = min(scores) - 0.01 if scores else 0.0
    min_fp = float('inf')
    thresholds = sorted(list(set(scores)))
    
    for t in thresholds:
        fp = sum(1 for s, l in zip(scores, labels) if s > t and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s <= t and l == 1)
        
        # Politique Zéro Défaut : on refuse tout seuil qui laisse passer une anomalie
        if fn == 0:
            if fp < min_fp:
                min_fp = fp
                best_thresh = t
                
    total_ok = sum(1 for l in labels if l == 0)
    faux_positifs_pct = (min_fp / total_ok) * 100 if total_ok > 0 else 0
    print(f"\n-> Politique Zéro Défaut validée (100% des défauts interceptés).")
    print(f"-> Taux de Faux Positifs (Pièces saines jetées à tort) : {faux_positifs_pct:.1f}% ({min_fp}/{total_ok})")
    
    return best_thresh
