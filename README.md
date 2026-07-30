# Eco-Tri : Classification de Déchets Recyclables

Application de classification d'images de déchets en 6 catégories :
**cardboard, glass, metal, paper, plastic, trash**

## Architecture du projet

Ce projet est divisé en deux parties :

### Partie 1 : Google Colab (Entraînement du modèle)
- **Fichier** : `colab/training.ipynb`
- **Rôle** : Chargement des données, entraînement du modèle, évaluation, génération des métriques et graphiques
- **Voir** : `colab/README_COLAB.md` pour les instructions détaillées

### Partie 2 : VSCode local (Application de démo)
- **Fichier** : `app/app.py`
- **Rôle** : Interface Streamlit pour tester le modèle sur de nouvelles images
- **Nécessite** : Le fichier `results/best_model.pth` généré par Colab

## Démarrage rapide

### Étape 1 : Entraîner le modèle sur Colab
1. Ouvrir `colab/training.ipynb` dans Google Colab
2. Exécuter toutes les cellules
3. Télécharger `best_model.pth` et les résultats dans le dossier `results/`

### Étape 2 : Lancer la démo en local
```bash
pip install -r requirements.txt
streamlit run app/app.py
