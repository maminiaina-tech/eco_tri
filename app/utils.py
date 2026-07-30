"""
Fonctions utilitaires pour l'application Streamlit.
Contient les transformations d'image et les informations sur les 5 classes.
"""
from PIL import Image
from torchvision import transforms

# Taille d'entrée du modèle (224x224 pour ResNet18)
IMG_SIZE = 224

# Transformations identiques à celles utilisées pendant l'entraînement (phase test)
predict_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Moyennes ImageNet
                         [0.229, 0.224, 0.225])   # Écarts-types ImageNet
])


def preprocess_image(image: Image.Image):
    """
    Prétraite une image PIL pour la prédiction.
    
    Args:
        image: image PIL (RGB ou autre)
    
    Returns:
        tenseur PyTorch avec dimension batch (1, 3, 224, 224)
    """
    image = image.convert('RGB')
    tensor = predict_transforms(image)
    return tensor.unsqueeze(0)


# Informations sur les 5 classes (avec touche locale malgache)
# L'ordre DOIT correspondre à l'ordre alphabétique utilisé par ImageFolder
CLASS_INFO = {
    'cardboard': {
        'fr': 'Carton',
        'mg': 'Kartôna',
        'conseil_fr': 'Pliez-le pour gagner de la place. Déposez-le dans la poubelle de tri.',
        'conseil_mg': 'Averino ho kely ary apetraho amin\'ny fako azo averina.',
    },
    'metal': {
        'fr': 'Métal',
        'mg': 'Vy',
        'conseil_fr': 'Les boîtes de conserve, canettes et aluminium vont au tri.',
        'conseil_mg': 'Ny boaty vy sy kanety dia azo averina.',
    },
    'paper': {
        'fr': 'Papier',
        'mg': 'Taratasy',
        'conseil_fr': 'Papier propre uniquement. Pas de papier souillé ou gras.',
        'conseil_mg': 'Taratasy madio ihany. Aza apetraka ny taratasy maloto.',
    },
    'plastic': {
        'fr': 'Plastique',
        'mg': 'Plastika',
        'conseil_fr': 'Rincez les bouteilles. Les bouchons peuvent rester.',
        'conseil_mg': 'Sasao ny tavoahangy. Ny tapony dia azo avela.',
    },
    'trash': {
        'fr': 'Déchet non recyclable',
        'mg': 'Fako tsy azo averina',
        'conseil_fr': 'Ce déchet ne peut pas être recyclé. Poubelle classique.',
        'conseil_mg': 'Ity fako ity dia tsy azo averina. Apetraho amin\'ny fako mahazatra.',
    }
}