import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Identity()
        self.classifier = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.base(x))
    
class TwoNN(BaseModel):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden, num_classes)

class OneNN(BaseModel):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, num_classes)
        )
        self.classifier = nn.Identity()