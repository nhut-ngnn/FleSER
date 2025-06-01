import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=7, fusion_method=None, alpha=None, dropout_rate=0.3):
        super(FlexibleMMSER, self).__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.alpha = alpha

        self.text_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
    def fuzzy_membership(self, x, method='sigmoid'):
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

        if method == 'sigmoid':
            clamped_x = torch.clamp(x, -4, 4) 
            return torch.sigmoid(clamped_x)
        elif method == 'tanh':
            clamped_x = torch.clamp(x, -2, 2)  
            return torch.tanh(clamped_x)
        elif method == 'linear':
            x = F.softplus(x)  
            normalized_x = torch.clamp(x, 0, 1)
            return normalized_x
        elif method == 'gaussian':
            mean = 0.5
            std = 0.15 
            normalized_x = torch.clamp(x, 0, 1)
            return torch.exp(-torch.pow(normalized_x - mean, 2) / (2 * std**2))
        elif method == 'piecewise':
            normalized_x = torch.clamp(x, 0, 1)
            return torch.where(
                normalized_x < 0.3, 0.2 * normalized_x,
                torch.where(normalized_x < 0.7, 0.5 + 0.5 * normalized_x, 0.9)
            )
        else:
            raise ValueError(f"Unknown fuzzy membership method: {method}")

    def select_fuzzy_type(self, input_data):
        mean_value = input_data.mean().item()
        std_dev = input_data.std().item()
        skewness = torch.mean((input_data - mean_value)**3) / (std_dev**3 + 1e-5)  
        min_val, max_val = input_data.min().item(), input_data.max().item()

        if -0.5 < skewness < 0.5 and 0 <= min_val < max_val <= 1:
            if std_dev < 0.1:
                return 'linear'
            else:
                return 'gaussian'
        elif skewness < -0.5 or skewness > 0.5: 
            return 'tanh'
        elif mean_value > 0.7 or max_val > 1.5: 
            return 'sigmoid'
        else:
            return 'piecewise'

    def fuzzy_fusion(self, text_fuzzy, audio_fuzzy):
        text_fuzzy = text_fuzzy.unsqueeze(1) 
        audio_fuzzy = audio_fuzzy.unsqueeze(1)
        if self.fusion_method == 'self_attention':
            weighted_input = self.alpha * audio_fuzzy + (1 - self.alpha) * text_fuzzy
            attn_output, _ = self.multihead_attention(
                query=weighted_input, key=weighted_input, value=weighted_input
            )
            return attn_output.mean(dim=1)

        elif self.fusion_method == 'cross_attention':
            attn_text_audio, _ = self.multihead_attention(
                query=text_fuzzy,
                key=audio_fuzzy,
                value=audio_fuzzy
            )
            attn_audio_text, _ = self.multihead_attention(
                query=audio_fuzzy,
                key=text_fuzzy,
                value=text_fuzzy
            )
            fused_feature = self.alpha * attn_text_audio + (1 - self.alpha) * attn_audio_text
            return fused_feature.mean(dim=1)
        elif self.fusion_method =='concat':
            return torch.cat([text_fuzzy, audio_fuzzy], dim=1).mean(dim=1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def forward(self, text_embed, audio_embed):
        text_proj = self.text_projection(text_embed)
        audio_proj = self.audio_projection(audio_embed)

        text_fuzzy_type = self.select_fuzzy_type(text_proj)
        audio_fuzzy_type = self.select_fuzzy_type(audio_proj)

        text_fuzzy = self.fuzzy_membership(text_proj, method=text_fuzzy_type)        
        audio_fuzzy = self.fuzzy_membership(audio_proj, method=audio_fuzzy_type)

        fused_fuzzy = self.fuzzy_fusion(text_fuzzy, audio_fuzzy)

        y_logits = self.fc(fused_fuzzy)
        y_softmax = self.softmax(y_logits)

        return y_logits, y_softmax