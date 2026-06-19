#!/bin/bash

TAILLE_PAQUET=100

echo "=== VÉRIFICATION DES FICHIERS À ENVOYER ==="

# On utilise une méthode d'extraction brute sans awk pour préserver les espaces et caractères spéciaux
mapfile -t TOUS_LES_FICHIERS < <(git status --porcelain | sed 's/^[ MAD?][ MAD?] //')
total_a_traiter=${#TOUS_LES_FICHIERS[@]}

echo "Nombre total de fichiers à envoyer : $total_a_traiter"

if [ "$total_a_traiter" -eq 0 ] || [ -z "${TOUS_LES_FICHIERS[0]}" ]; then
    echo "Aucun fichier à pousser. Tout est déjà sur GitHub !"
    exit 0
fi

echo "=== DÉBUT DU PUSH PAR PAQUETS SÉCURISÉ ==="

compteur=0

for ((i=0; i<total_a_traiter; i+=TAILLE_PAQUET)); do
    paquet=("${TOUS_LES_FICHIERS[@]:i:TAILLE_PAQUET}")
    
    compteur=$((compteur + ${#paquet[@]}))
    pourcentage=$(( (compteur * 100) / total_a_traiter ))
    
    echo "[${pourcentage}%] Préparation du paquet ($compteur/$total_a_traiter)..."
    
    # On ajoute les fichiers du paquet un par un pour éviter qu'un seul fichier manquant ne bloque tout le reste
    for file in "${paquet[@]}"; do
        if [ -f "$file" ]; then
            git add "$file"
        fi
    done
    
    # On vérifie s'il y a effectivement quelque chose de prêt à être commité
    if ! git diff --cached --quiet; then
        echo "   -> Envoi du paquet de fichiers sur GitHub..."
        git commit -m "Push groupe automatique : $compteur/$total_a_traiter" --quiet
        git push origin main
        echo "--------------------------------------------------"
    else
        echo "   -> [Info] Paquet vide ou fichiers déjà indexés, passage au suivant."
    fi
done

echo "=== FIN DU PROCESSUS : TOUT EST SUR GITHUB ! ==="
