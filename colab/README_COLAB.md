# Instructions pour Google Colab

## Objectif
Ce notebook entraîne le modèle de classification de déchets et génère tous les résultats (modèle, métriques, graphiques).

## Étapes à suivre

### 1. Ouvrir le notebook sur Colab
- Aller sur https://colab.research.google.com/drive/1I44SP3TgbOsufmVtG03jenPQQ6iLEM2e


### 2. Configurer le runtime
- Menu : **Exécuter** → **Modifier le type d'exécution**
- Choisir : **GPU** (T4 gratuit)
- Cliquer sur **Enregistrer**

### 3. Connecter Google Drive (optionnel mais recommandé)
Pour sauvegarder les résultats automatiquement :
- Exécuter la cellule de montage Drive
- Les résultats seront dans `MonDrive/EcoTri/results/`

### 4. Télécharger le dataset
Le notebook télécharge automatiquement le dataset Kaggle.
**Important** : Si vous utilisez l'API Kaggle, configurez votre `kaggle.json` (voir dans le notebook).

Alternative simple : Télécharger manuellement le dataset depuis :
https://www.kaggle.com/datasets/vipriti/better-dataset-trashnet
Puis l'uploader dans Colab.

### 5. Exécuter le notebook
- Menu : **Exécuter** → **Exécuter toutes les cellules**
- Durée : ~15-20 minutes avec GPU T4

### 6. Récupérer les résultats
À la fin du notebook, téléchargez ces fichiers :
- `best_model.pth` → à placer dans `results/`
- `confusion_matrix.png` → à placer dans `results/`
- `training_curves.png` → à placer dans `results/`
- `metrics_report.txt` → à placer dans `results/`

### 7. Lancer la démo locale
Une fois les fichiers dans `results/`, lancez :
```bash
streamlit run app/app.py