# Neural Edge-Art

Edge-aware neural style transfer using VGG19 features and multiple edge detection methods. Apply artistic styles to photographs while preserving structural edges.

## Features

- **Multiple Edge Detection**: Canny, Sobel, Laplacian, Prewitt operators
- **VGG19 Feature Extraction**: Pre-trained ImageNet model for perceptual loss
- **Gram Matrix Style Loss**: Captures artistic texture patterns
- **Edge Preservation Loss**: Maintains structural integrity
- **Creative Post-Processing**: Sketch, Neon Glow, Watercolor, Edge Overlay filters
- **Gradio Interface**: Interactive web UI for experimentation
- **GPU Optimized**: Supports T4, V100, A100 with automatic memory management

## Installation

pip install torch torchvision opencv-python-headless gradio lpips matplotlib pillow numpy

## Quick Start

### Basic Usage

import torch
from neural_edge_art import NeuralEdgeArt

# Initialize (adjust image_size based on GPU)
art = NeuralEdgeArt(image_size=512)

# Perform style transfer
result, loss_history, edges = art.style_transfer(
    content_img="photo.jpg",
    style_img="painting.jpg",
    edge_method='canny',
    num_steps=200,
    content_weight=1.0,
    style_weight=1e6,
    edge_weight=100
)

# Save result
result.save("stylized_output.png")

## Core Components

### 1. Edge Detection Module

import cv2
import numpy as np

class EdgeAwareProcessor:
    def __init__(self):
        self.kernels = {
            'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
            'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
            'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
            'prewitt_x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
            'prewitt_y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        }
    
    def extract_edges(self, image, method='canny', threshold1=50, threshold2=150):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        if method == 'canny':
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method == 'sobel':
            sobel_x = cv2.filter2D(gray, -1, self.kernels['sobel_x'])
            sobel_y = cv2.filter2D(gray, -1, self.kernels['sobel_y'])
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        elif method == 'laplacian':
            edges = cv2.filter2D(gray, -1, self.kernels['laplacian'])
            edges = np.abs(edges)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        elif method == 'prewitt':
            prewitt_x = cv2.filter2D(gray, -1, self.kernels['prewitt_x'])
            prewitt_y = cv2.filter2D(gray, -1, self.kernels['prewitt_y'])
            edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        
        return edges

### 2. VGG19 Feature Extractor

import torch.nn as nn
import torchvision.models as models

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract 4 layers for multi-scale features
        self.layer1 = vgg[:4]   # conv1_2
        self.layer2 = vgg[4:9]  # conv2_2
        self.layer3 = vgg[9:18] # conv3_4
        self.layer4 = vgg[18:27] # conv4_4
    
    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        return feat1, feat2, feat3, feat4

### 3. Loss Functions

import torch.nn.functional as F

class StyleTransferLoss:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.mse = nn.MSELoss()
    
    def gram_matrix(self, features):
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def content_loss(self, generated_features, content_features):
        return self.mse(generated_features, content_features)
    
    def style_loss(self, generated_features, style_features):
        loss = 0
        for gen_feat, style_feat in zip(generated_features, style_features):
            gen_gram = self.gram_matrix(gen_feat)
            style_gram = self.gram_matrix(style_feat)
            loss += self.mse(gen_gram, style_gram)
        return loss
    
    def edge_loss(self, generated_img, content_edges):
        # Convert to grayscale
        gray = 0.299 * generated_img[:, 0] + 0.587 * generated_img[:, 1] + 0.114 * generated_img[:, 2]
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32).view(1, 1, 3, 3).to(device)
        
        gray = gray.unsqueeze(1)
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        generated_edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize
        generated_edges = (generated_edges - generated_edges.min()) / \
                         (generated_edges.max() - generated_edges.min() + 1e-8)
        
        return self.mse(generated_edges, content_edges)
    
    def total_loss(self, generated_img, content_img, style_img, content_edges,
                   content_weight=1.0, style_weight=1e6, edge_weight=100):
        gen_features = self.feature_extractor(generated_img)
        content_features = self.feature_extractor(content_img)[3]
        style_features = self.feature_extractor(style_img)
        
        c_loss = self.content_loss(gen_features[3], content_features)
        s_loss = self.style_loss(gen_features, style_features)
        e_loss = self.edge_loss(generated_img, content_edges)
        
        total = content_weight * c_loss + style_weight * s_loss + edge_weight * e_loss
        return total, c_loss, s_loss, e_loss

## Loss Function

**Total Loss:**
