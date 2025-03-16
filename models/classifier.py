import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Any


class ArtClassifier(nn.Module):
    """Neural network for classifying AI vs human-made art using a pretrained backbone"""
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1, pretrained: bool = True):
        """
        Args:
            model_name: Name of the pretrained model to use as backbone
            num_classes: Number of output classes (1 for binary classification with sigmoid)
            pretrained: Whether to use pretrained weights
        """
        super(ArtClassifier, self).__init__()
        
        # Initialize the pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remove the classifier
        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()  # Remove the classification head
        else:
            raise ValueError(f"Model {model_name} not supported")
            
        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # For binary classification, we'll use BCEWithLogitsLoss instead of applying sigmoid here
        # This is safer with mixed precision training
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits  # Return logits directly (no sigmoid)
    
    
def create_model(config: Dict[str, Any]) -> ArtClassifier:
    """
    Create a model instance based on configuration
    
    Args:
        config: Dictionary containing model configuration
        
    Returns:
        model: Instantiated model
    """
    model = ArtClassifier(
        model_name=config.get('model_name', 'resnet50'),
        num_classes=config.get('num_classes', 1),
        pretrained=config.get('pretrained', True)
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model