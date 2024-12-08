import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=4, fusion_method='weighted'):
        super(FlexibleMMSER, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(.2)
        self.linear = nn.Linear(768*2, 512)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.fusion_method = fusion_method
        self.alpha = 0.3
        
        self.attention = nn.Linear(768 * 2, 1)

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
            attention_weights = torch.sigmoid(self.attention(concat_embed))
            fused_embed = attention_weights * text_fuzzy + (1 - attention_weights) * audio_fuzzy
            return fused_embed
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def forward(self, text_embed, audio_embed):
        text_fuzzy = self.fuzzy_membership(text_embed)
        audio_fuzzy = self.fuzzy_membership(audio_embed)
        
        fused_fuzzy = self.fuzzy_fusion(text_fuzzy, audio_fuzzy) 
        
        concat_embed = torch.cat((text_embed, audio_embed), dim=1) 

        x = self.linear(concat_embed) 
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear1(x) 
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) 
        x = F.relu(x)
        y_logits = self.linear3(x)  
        
        y_softmax = self.softmax(y_logits)
        
        return y_logits, y_softmax
