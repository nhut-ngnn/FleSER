import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=4, fusion_method='attention', dropout_rate=0.3):
        super(FlexibleMMSER, self).__init__()
        
        # Model parameters
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.alpha = 0.3
        
        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.multihead_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def fuzzy_membership(self, x):
        """Apply a fuzzy membership function."""
        return torch.sigmoid(x)

    def fuzzy_fusion(self, text_fuzzy, audio_fuzzy):
        """Perform fuzzy fusion based on the specified method."""
        if self.fusion_method == 'weighted':
            return self.alpha * text_fuzzy + (1 - self.alpha) * audio_fuzzy
        elif self.fusion_method == 'min':
            return torch.min(text_fuzzy, audio_fuzzy)
        elif self.fusion_method == 'max':
            return torch.max(text_fuzzy, audio_fuzzy)
        elif self.fusion_method == 'product':
            return text_fuzzy * audio_fuzzy
        elif self.fusion_method == 'sum':
            return text_fuzzy + audio_fuzzy - text_fuzzy * audio_fuzzy
        elif self.fusion_method == 'attention':
            concat_embed = torch.cat((text_fuzzy, audio_fuzzy), dim=1)
            attention_weights = self.attention(concat_embed)
            return attention_weights * text_fuzzy + (1 - attention_weights) * audio_fuzzy
        elif self.fusion_method == 'MHA':
            text_fuzzy = text_fuzzy.unsqueeze(1)
            audio_fuzzy = audio_fuzzy.unsqueeze(1)
            attn_output, _ = self.multihead_attention(text_fuzzy, audio_fuzzy, audio_fuzzy)
            fused_output = self.fuzzy_membership(attn_output.mean(dim=1))
            return fused_output
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def forward(self, text_embed, audio_embed):
        text_proj = self.text_projection(text_embed)
        audio_proj = self.audio_projection(audio_embed)
        
        text_fuzzy = self.fuzzy_membership(text_proj)
        audio_fuzzy = self.fuzzy_membership(audio_proj)

        fused_fuzzy = self.fuzzy_fusion(text_fuzzy, audio_fuzzy)

        concat_embed = torch.cat((text_proj, audio_proj), dim=1)

        y_logits = self.fc(concat_embed)

        y_softmax = self.softmax(y_logits)
        
        return y_logits, y_softmax
