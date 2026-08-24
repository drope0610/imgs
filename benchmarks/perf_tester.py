import time
import torch
from anomalib.models import EfficientAd

def benchmark_model(device="cuda"):
    print("="*50)
    print("⚡ BENCHMARK DE PERFORMANCE : EFFICIENT-AD ⚡")
    print("="*50)

    # Chargement du modèle en mode évaluation (désactive l'entraînement)
    model = EfficientAd().to(device)
    model.eval()

    # Création d'une fausse image (tenseur) au format standard (1 batch, 3 couleurs, 256x256)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    print("🔥 Préchauffage du GPU (Warmup)...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    print("⏱️ Lancement du test sur 500 itérations...")
    iterations = 500
    start_time = time.time()

    # Boucle d'inférence pure
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    # Calculs des métriques
    end_time = time.time()
    total_time = end_time - start_time
    fps = iterations / total_time
    latency = (total_time / iterations) * 1000

    print("\n📊 RÉSULTATS DU BENCHMARK :")
    print(f"➜ Matériel : {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"➜ Débit (FPS) : {fps:.2f} images/seconde")
    print(f"➜ Latence moyenne : {latency:.2f} ms / image")
    print("="*50)

if __name__ == "__main__":
    # Détection automatique du GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark_model(device)
