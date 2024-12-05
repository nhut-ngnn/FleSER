import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FuzzyLayer, self).__init__()
        # Fuzzy membership functions parameters
        self.mu_centers = nn.Parameter(torch.randn(num_classes, input_dim))
        self.mu_sigmas = nn.Parameter(torch.rand(num_classes, input_dim))
    
    def gaussian_membership(self, x, center, sigma):
        """
        Gaussian membership function
        Calculates degree of membership using Gaussian distribution
        """
        return torch.exp(-((x - center) ** 2) / (2 * (sigma ** 2)))
    
    def forward(self, x):
        """
        Compute fuzzy membership degrees for each class
        """
        # Compute membership degrees for each feature dimension
        memberships = []
        for i in range(self.mu_centers.shape[0]):
            class_membership = self.gaussian_membership(x, self.mu_centers[i], self.mu_sigmas[i])
            memberships.append(class_membership)
        
        # Aggregate membership degrees
        fuzzy_output = torch.stack(memberships, dim=1)
        return fuzzy_output

class FlexibleFuzzyMMSER(nn.Module):
    def __init__(self, text_embed_dim=480, audio_embed_dim=480, num_classes=4, dropout_rate=0.3):
        super(FlexibleFuzzyMMSER, self).__init__()
        
        # Concatenation dimension
        concat_embed_dim = text_embed_dim + audio_embed_dim
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(concat_embed_dim, 960),
            nn.BatchNorm1d(960),
            nn.GELU()
        )
        
        # Fuzzy Layer
        self.fuzzy_layer = FuzzyLayer(input_dim=960, num_classes=num_classes)
        
        # Dropout and linear layers
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(960, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, num_classes)
        
        # Softmax for final classification
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, text_embed, audio_embed):
        # Concatenate text and audio embeddings
        concat_embed = torch.cat((text_embed, audio_embed), dim=1)
        
        # Project embeddings
        projected_embed = self.projection(concat_embed)
        
        # Apply dropout
        x = self.dropout(projected_embed)
        
        # Fuzzy membership computation
        fuzzy_memberships = self.fuzzy_layer(x)
        
        # Continue with standard neural network layers
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        y_logits = self.linear3(x)
        
        # Combine logits with fuzzy memberships (optional weighted approach)
        weighted_logits = y_logits * fuzzy_memberships
        
        # Softmax for final probabilities
        y_softmax = self.softmax(weighted_logits)
        
        return y_logits, y_softmax, fuzzy_memberships
