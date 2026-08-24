#!/bin/bash
echo "Attente de la fin de l'entraînement sur jetson2..."

# Boucle infinie : vérifie toutes les 2 minutes si le modèle final est généré
while ! ssh -o BatchMode=yes jetson2 "ls ~/trained_models/pillQC250ep.pt" >/dev/null 2>&1; do
    sleep 120
done

echo "Modèle généré ! Téléchargement en cours..."
# Récupération du dossier
scp -r jetson2:~/trained_models /Users/pedro/Desktop/Inria/imgs/
echo "Dossier rapatrié avec succès sur la machine locale."
