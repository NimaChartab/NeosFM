# NEOS Foundation Model

This repository contains a foundation model for the NEO Surveyor mission. The model processes astronomical difference images (NC1 and NC2) along with tabular features to create learned embeddings for similarity search and object classification.

## Key Features

- **Dual-Channel Image Processing**: Single ResNet processes NC1 and NC2 images simultaneously as 2-channel input
- **Tabular Processing**: 50 selected features
- **Multimodal Fusion**: Cross-attention mechanism combines image and tabular representations
- **Interactive Exploration**: Jupyter notebook interface for similarity search and visualization
- **UMAP Visualization**: 2D embedding space visualization for intuitive exploration

## Model Architecture

```
Input: NC1 Image (61x61) + NC2 Image (61x61) + Tabular Features (50D)
       ↓
Dual-Channel ResNet (NC1+NC2 → 512D) + Tabular MLP (50D → 512D)
       ↓
Cross-Attention Fusion
       ↓
Output: 512D Normalized Embeddings
```

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install torch numpy matplotlib pandas scikit-learn umap-learn astropy ipywidgets jupyter
```

## Model Files

The trained model weights (`best_model.pt`) are included in this repository. The pre-computed embeddings cache (`model_embeddings.npz`) is excluded due to size limits but will be automatically generated when you first run the notebook.

## Usage

### Quick Start

1. Open the Jupyter notebook:
```bash
jupyter notebook neos_explorer.ipynb
```

2. Run all cells to load the model and start the interactive interface

### Interactive Features

- **Search Settings**: Adjust number of similar objects to display (1-20)
- **Custom Search**: Search for specific source IDs by entering the ID
- **Quality Filters**: Filter objects by NC2snr, NC2snrpm, and NC2mag ranges
- **UMAP Region Selection**: Select X/Y coordinate ranges in 2D embedding space
- **Quick Search Buttons**: 
  - Random Search: Find random objects from filtered region
  - Random Real: Find random real objects from filtered region  
  - Random Fake: Find random fake objects from filtered region
- **Similarity Display**: Shows query object + most similar objects with images and metadata


![NEOS Explorer Interface](figures/interface_example.png)


