"""
NEO Surveyor Foundation Model
============================

Dual-channel ResNet foundation model for processing astronomical images and tabular data.
Combines NC1 and NC2 images as 2-channel input with tabular features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResNetBlock(nn.Module):
    """ResNet basic block with residual connections"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class DualChannelResNetEncoder(nn.Module):
    """Single ResNet encoder for dual-channel (NC1+NC2) astronomical images"""
    
    def __init__(self, embed_dim=512, blocks_per_layer=[2, 2, 2, 2]):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Initial convolution for 2-channel input (NC1 + NC2)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, blocks_per_layer[0], stride=1)
        self.layer2 = self._make_layer(64, 128, blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks_per_layer[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks_per_layer[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, nc1_nc2_combined):
        # Input: (batch_size, 2, 61, 61) - NC1 and NC2 as channels
        x = self.conv1(nc1_nc2_combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        
        embeddings = self.projection(x)
        
        # L2 normalize for contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class TabularEncoder(nn.Module):
    """Tabular encoder for astronomical features with proper normalization"""
    
    def __init__(self, input_dim=50, embed_dim=512, hidden_dims=[256, 256, 128]):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        
        # Input projection with batch normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(self._make_residual_block(hidden_dims[i], hidden_dims[i+1]))
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def _make_residual_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
    def forward(self, x):
        # Input projection
        out = self.input_proj(x)
        
        # Residual blocks with skip connections
        for i, block in enumerate(self.blocks):
            identity = out
            out = block(out)
            
            # Skip connection if dimensions match
            if identity.shape[-1] == out.shape[-1]:
                out = out + identity
            out = F.relu(out)
        
        # Final projection
        embeddings = self.final_proj(out)
        
        # L2 normalize for contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class NEOSFoundationModel(nn.Module):
    """NEOS foundation model with dual-channel ResNet and tabular encoder"""
    
    def __init__(self, tabular_dim=50, embed_dim=512):
        super().__init__()
        
        # Single ResNet for dual-channel images (NC1+NC2)
        self.dual_channel_encoder = DualChannelResNetEncoder(embed_dim=embed_dim)
        
        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim, embed_dim=embed_dim
        )
        
        # Cross-modal attention fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Final fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, nc1_images, nc2_images, tabular_data, return_individual=False):
        # Combine NC1 and NC2 into 2-channel input
        dual_channel_input = torch.cat([nc1_images, nc2_images], dim=1)
        
        # Single ResNet processes both channels
        image_embeds = self.dual_channel_encoder(dual_channel_input)
        
        # Tabular encoder
        tabular_embeds = self.tabular_encoder(tabular_data)
        
        # Cross-modal attention between image and tabular
        modal_stack = torch.stack([image_embeds, tabular_embeds], dim=1)
        attended_modals, _ = self.cross_attention(modal_stack, modal_stack, modal_stack)
        attended_image, attended_tabular = attended_modals[:, 0], attended_modals[:, 1]
        
        # Final fusion
        combined = torch.cat([attended_image, attended_tabular], dim=1)
        fused_embeds = self.fusion(combined)
        fused_embeds = F.normalize(fused_embeds, p=2, dim=1)
        
        if return_individual:
            return image_embeds, tabular_embeds, fused_embeds
        return fused_embeds


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for multimodal learning"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings1, embeddings2, embeddings3=None):
        batch_size = embeddings1.size(0)
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=embeddings1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
