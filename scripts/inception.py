import torch.nn as nn
import torchmetrics
from torchmetrics import Accuracy

class InceptionCustom(nn.Module):
    def __init__(
        self,
        num_classes
    ) -> None:
        super(InceptionCustom, self).__init__()
        self.model =  torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)