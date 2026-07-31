"""
Eco-Tri : Application de démonstration Streamlit
Classification de déchets recyclables (5 classes)

Lancer avec : streamlit run app/app.py
"""
import os
import torch
import streamlit as st
from PIL import Image

from model import load_trained_model
from utils import preprocess_image, CLASS_INFO

# Configuration de la page
st.set_page_config(
    page_title="Eco-Tri - Classification de Déchets",
    page_icon="🌍",
    layout="wide"
)

# Classes du modèle (ordre alphabétique, identique à ImageFolder)
CLASS_NAMES = list(CLASS_INFO.keys())


@st.cache_resource
def load_model():
    """Charge le modèle une seule fois (mis en cache par Streamlit)."""
    model_path = 'results/best_model.pth'
    
    if not os.path.exists(model_path):
        st.error(f"Modèle non trouvé : {model_path}")
        st.info("""
        **Veuillez d'abord :**
        1. Exécuter le notebook `colab/training.ipynb` sur Google Colab
        2. Télécharger `best_model.pth`
        3. Le placer dans le dossier `results/`
        """)
        return None
    
    device = torch.device('cpu')
    return load_trained_model(model_path, num_classes=len(CLASS_NAMES), device=device)


def predict(model, image):
    """Fait la prédiction sur une image PIL."""
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    return probabilities.numpy()


def main():
    # Titre principal
    st.title("Eco-Tri : Classification de Déchets Recyclables")
    st.markdown("""
    **Application de classification d'images de déchets**  
    Développée dans le cadre du projet M2 - 2026  
    Modèle : ResNet18 avec Transfer Learning (PyTorch)
    """)
    
    # Charger le modèle
    model = load_model()
    if model is None:
        st.stop()
    
    # ========== SIDEBAR ==========
    st.sidebar.markdown("### Langue / Fiteny")
    lang = st.sidebar.radio(
        "Langue", 
        ['Français', 'Malagasy'], 
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### À propos du modèle")
    st.sidebar.markdown(f"""
    - **Architecture** : ResNet18 (Transfer Learning)
    - **Classes** : {len(CLASS_NAMES)} catégories
    - **Framework** : PyTorch
    - **Dataset** : Garbage Classification (Kaggle)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Fichiers disponibles")
    results_dir = 'results'
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        for f in files:
            if not f.startswith('.'):
                st.sidebar.text(f" {f}")
    else:
        st.sidebar.warning("Dossier results/ non trouvé")
    
    # ========== ZONE PRINCIPALE ==========
    st.markdown("### Téléverser une image de déchet")
    uploaded_file = st.file_uploader(
        "Choisissez une image (JPG, PNG)...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Image uploadée', width='stretch')
        
        # Prédiction
        with st.spinner("Analyse en cours..."):
            probs = predict(model, image)
            pred_idx = probs.argmax()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100
        
        with col2:
            st.markdown("### Résultat")
            info = CLASS_INFO[pred_class]
            
            if lang == 'Français':
                st.success(f"**{info['fr']}**")
                st.metric("Confiance", f"{confidence:.1f}%")
                st.info(f"**Conseil de tri :** {info['conseil_fr']}")
            else:
                st.success(f"**{info['mg']}**")
                st.metric("Fahatokiana", f"{confidence:.1f}%")
                st.info(f"**Torohevitra :** {info['conseil_mg']}")
            
            # Graphique des probabilités
            st.markdown("### Probabilités par classe")
            prob_dict = {}
            for i, class_name in enumerate(CLASS_NAMES):
                icon = CLASS_INFO[class_name]['fr'].split(' ')[0]
                prob_dict[f"{icon} {class_name}"] = float(probs[i])
            st.bar_chart(prob_dict)
        
        # Top 3 prédictions
        st.markdown("### Top 3 des prédictions")
        top3_idx = probs.argsort()[-3:][::-1]
        cols = st.columns(3)
        for i, idx in enumerate(top3_idx):
            with cols[i]:
                class_name = CLASS_NAMES[idx]
                info = CLASS_INFO[class_name]
                st.markdown(f"**{info['fr']}**")
                st.progress(float(probs[idx]))
                st.caption(f"{probs[idx]*100:.1f}%")
    
    # ========== SECTION RÉSULTATS DU MODÈLE ==========
    st.markdown("---")
    st.markdown("### Résultats du modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        curves_path = 'results/training_curves.png'
        if os.path.exists(curves_path):
            st.image(curves_path, caption='Courbes d\'apprentissage', width='stretch')
        else:
            st.info("Courbes d'apprentissage non disponibles")
    
    with col2:
        cm_path = 'results/confusion_matrix.png'
        if os.path.exists(cm_path):
            st.image(cm_path, caption='Matrice de confusion', width='stretch')
        else:
            st.info("Matrice de confusion non disponible")
    
    # Rapport de métriques
    metrics_path = 'results/metrics_report.txt'
    if os.path.exists(metrics_path):
        st.markdown("### Rapport de métriques")
        with open(metrics_path, 'r', encoding='utf-8') as f:
            st.code(f.read(), language='text')
    
    # Footer
    st.markdown("---")
    st.markdown("""
                    **Un geste simple pour la planète !**  
    """)


if __name__ == "__main__":
    main()