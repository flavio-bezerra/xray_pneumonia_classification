import torch
import torch.nn as nn
from torchvision import models

class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=1, freeze_backbone=True):
        """
        Modelo EfficientNetB0 para classificação binária.
        """
        super(PneumoniaClassifier, self).__init__()
        
        # Carregar EfficientNetB0 pré-treinada
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Congelar pesos do backbone se solicitado
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Substituir o classificador original
        # O classifier original é um Sequential(Dropout, Linear)
        # Vamos manter a estrutura mas ajustar para nossa saída
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

    def unfreeze(self):
        """Descongela todos os parâmetros para fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
