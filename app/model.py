"""
Architecture du modèle de classification.
Utilise Transfer Learning avec ResNet18 pré-entraîné sur ImageNet.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes, freeze_base=False):
    """
    Crée un modèle ResNet18 avec Transfer Learning.
    
    Args:
        num_classes: nombre de classes de sortie
        freeze_base: si True, gèle les poids de la base
    
    Returns:
        modèle PyTorch
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model


def load_trained_model(model_path, num_classes, device='cpu'):
    """
    Charge un modèle entraîné depuis un fichier .pth.
    
    Args:
        model_path: chemin vers le fichier .pth
        num_classes: nombre de classes
        device: 'cpu' ou 'cuda'
    
    Returns:
        modèle PyTorch en mode évaluation
    """
    model = create_model(num_classes, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model