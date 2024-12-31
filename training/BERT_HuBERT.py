import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=4, fusion_method=None, alpha=None, dropout_rate=0.3):
        super(FlexibleMMSER, self).__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.alpha = alpha

        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.multihead_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

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

    def fuzzy_membership(self, x, method='sigmoid'):
        if method == 'sigmoid':
            x = torch.clamp(x, -6, 6)
            return torch.sigmoid(x)
        elif method == 'tanh':
            x = torch.clamp(x, -3, 3)
            return torch.tanh(x)
        elif method == 'linear':
            x = F.relu(x)
            x = torch.clamp(x, 0, 1)
            return x
        elif method == 'gaussian':
            mean = 0.5
            std = 0.1
            x = torch.clamp(x, 0, 1)
            return torch.exp(-torch.pow(x - mean, 2) / (2 * std**2))
        elif method == 'piecewise':
            x = torch.clamp(x, 0, 1)
            return torch.where(
                x < 0.3, 0.2 * x,
                torch.where(x < 0.7, 0.5 + 0.5 * x, 0.9)
            )
        else:
            raise ValueError(f"Unknown fuzzy membership method: {method}")

    def select_fuzzy_type(self, input_data):
        mean_value = input_data.mean().item()
        std_dev = input_data.std().item()
        min_val, max_val = input_data.min().item(), input_data.max().item()

        if min_val < 0 and max_val > 0:  
            return 'tanh'
        elif 0 <= min_val < max_val <= 1:  
            if std_dev < 0.1:  
                return 'linear'
            else:  
                return 'gaussian'
        elif mean_value > 1 or std_dev > 1: 
            return 'sigmoid'
        elif 0 <= min_val < max_val:  
            return 'piecewise'
        else:
            return 'sigmoid' 

    def fuzzy_fusion(self, text_fuzzy, audio_fuzzy):
        text_fuzzy = text_fuzzy.unsqueeze(1) 
        audio_fuzzy = audio_fuzzy.unsqueeze(1)
        if self.fusion_method == 'self_attention':
            weighted_input = self.alpha * text_fuzzy + (1 - self.alpha) * audio_fuzzy  
            concat_embed = torch.cat((text_fuzzy, audio_fuzzy), dim=1) 
            attn_output, _ = self.multihead_attention(query=weighted_input, key=concat_embed, value=concat_embed)
            fused_output = attn_output.mean(dim=1)  
            return self.fuzzy_membership(fused_output)
        elif self.fusion_method == 'cross_attention':
            weighted_input = self.alpha * text_fuzzy + (1 - self.alpha) * audio_fuzzy
            weighted_input = weighted_input.unsqueeze(1)
            attn_output, _ = self.multihead_attention(query=audio_fuzzy, key=text_fuzzy, value=text_fuzzy)
            fused_output = attn_output.mean(dim=1)  
            return self.fuzzy_membership(fused_output)
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
        concat_embed = torch.cat((text_proj, audio_proj), dim=1)
        y_logits = self.fc(concat_embed)
        y_softmax = self.softmax(y_logits)

        return y_logits, y_softmax
