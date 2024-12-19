import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=4, fusion_method='concat', dropout_rate=0.3):
        super(FlexibleMMSER, self).__init__()
        
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # Projection layers for text and audio embeddings
        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def fuzzy_membership(self, x):
        """Apply a fuzzy membership function."""
        return torch.sigmoid(x)

    def forward(self, text_embed, audio_embed):
        # Apply projection layers
        text_proj = self.text_projection(text_embed)
        audio_proj = self.audio_projection(audio_embed)
        
        # Apply fuzzy membership
        text_fuzzy = self.fuzzy_membership(text_proj)
        audio_fuzzy = self.fuzzy_membership(audio_proj)
        
        # Fusion (simple concatenation of fuzzy embeddings)
        if self.fusion_method == 'concat':
            fused_embed = torch.cat((text_fuzzy, audio_fuzzy), dim=1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Classification
        y_logits = self.fc(fused_embed)
        y_softmax = self.softmax(y_logits)
        
        return y_logits, y_softmax
