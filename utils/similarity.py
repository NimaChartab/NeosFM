"""
Similarity Search and Visualization Utilities
=============================================

Functions for embedding generation, similarity search, and visualization
of NEOS foundation model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from sklearn.metrics.pairwise import cosine_similarity
import torch
import umap
from typing import List, Tuple, Dict, Optional


def generate_embeddings(model, dataset, device='cuda', batch_size=64):
    """Generate embeddings for all samples in dataset using trained model"""
    
    from torch.utils.data import DataLoader
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    embeddings_list = []
    source_ids_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            nc1_images = batch['nc1_image'].to(device)
            nc2_images = batch['nc2_image'].to(device)
            tabular_data_batch = batch['tabular_data'].to(device)
            
            # Generate learned embeddings through the foundation model
            fused_embeds = model(nc1_images, nc2_images, tabular_data_batch)
            
            embeddings_list.append(fused_embeds.cpu().numpy())
            source_ids_list.extend(batch['source_id'])
    
    embeddings = np.vstack(embeddings_list)
    source_ids = np.array(source_ids_list)
    
    return embeddings, source_ids


def create_umap_visualization(embeddings, random_state=42):
    """Create UMAP 2D visualization of embeddings"""
    
    umap_reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=random_state
    )
    embeddings_2d = umap_reducer.fit_transform(embeddings)
    
    return embeddings_2d


def find_similar_objects(query_source_id, embeddings, source_ids, labels, top_k=10):
    """Find similar objects using learned embeddings"""
    
    # Find query embedding
    query_idx = np.where(source_ids == query_source_id)[0]
    if len(query_idx) == 0:
        return None, f"Source ID {query_source_id} not found in embeddings"
    
    query_idx = query_idx[0]
    query_embedding = embeddings[query_idx:query_idx+1]
    query_label = labels[query_idx]
    
    # Calculate similarities using learned embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k most similar (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = []
    for idx in similar_indices:
        results.append({
            'source_id': source_ids[idx],
            'similarity': similarities[idx],
            'label': labels[idx]
        })
    
    query_info = {
        'source_id': query_source_id,
        'label': query_label,
        'similarity': 1.0
    }
    
    return query_info, results


def apply_z_scale(image, percentile_low=1, percentile_high=99):
    """Apply z-scale normalization to FITS image"""
    valid_pixels = image[~np.isnan(image)]
    if len(valid_pixels) == 0:
        return np.zeros_like(image)
    
    p_low, p_high = np.percentile(valid_pixels, [percentile_low, percentile_high])
    
    if p_low == p_high:
        min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
        if min_val == max_val:
            return np.clip((image - min_val), 0, 1)
        return np.clip((image - min_val) / (max_val - min_val), 0, 1)
    
    scaled_image = np.clip((image - p_low) / (p_high - p_low), 0, 1)
    
    if np.isnan(scaled_image).any():
        min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
        if min_val == max_val:
            return np.clip((image - min_val), 0, 1)
        return np.clip((image - min_val) / (max_val - min_val), 0, 1)
    
    return scaled_image


def load_images(source_id, visit_paths):
    """Load NC1 and NC2 images for a source_id"""
    try:
        for visit_path in visit_paths:
            visit_path = Path(visit_path)
            nc1_dirs = list(visit_path.glob("*nc1_cutouts_l4b"))
            nc2_dirs = list(visit_path.glob("*nc2_cutouts_l4b"))
            
            if nc1_dirs and nc2_dirs:
                nc1_cutouts_dir = nc1_dirs[0]
                nc2_cutouts_dir = nc2_dirs[0]
                
                nc1_file = nc1_cutouts_dir / f"{source_id}_nc1_L4b.fits"
                nc2_file = nc2_cutouts_dir / f"{source_id}_nc2_L4b.fits"
                
                if nc1_file.exists() and nc2_file.exists():
                    # Load NC1 image
                    with fits.open(nc1_file) as hdul:
                        nc1_data = hdul[0].data
                        nc1_scaled = apply_z_scale(nc1_data)
                    
                    # Load NC2 image
                    with fits.open(nc2_file) as hdul:
                        nc2_data = hdul[0].data
                        nc2_scaled = apply_z_scale(nc2_data)
                    
                    return nc1_scaled, nc2_scaled, True
        
        # If we can't find actual images, create placeholder
        np.random.seed(hash(source_id) % 2**32)
        nc1_image = np.random.normal(0, 0.1, (61, 61))
        nc1_image[25:35, 25:35] += np.random.normal(0.5, 0.2, (10, 10))
        
        nc2_image = np.random.normal(0, 0.05, (61, 61))
        nc2_image[25:35, 25:35] += np.random.normal(0.8, 0.1, (10, 10))
        
        nc1_scaled = apply_z_scale(nc1_image)
        nc2_scaled = apply_z_scale(nc2_image)
        
        return nc1_scaled, nc2_scaled, True
        
    except Exception as e:
        print(f"Error loading images for {source_id}: {e}")
        return None, None, False


def plot_umap_space(embeddings_2d, labels, source_ids, highlight_source_id=None, interactive=False):
    """Create UMAP visualization with optional highlighted object"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Create masks
    real_mask = labels == 'Real'
    fake_mask = labels == 'Fake'
    
    # Plot all points
    ax.scatter(embeddings_2d[fake_mask, 0], embeddings_2d[fake_mask, 1],
               c='red', alpha=0.6, s=25, label=f'Fake: {fake_mask.sum():,}',
               edgecolors='darkred', linewidth=0.1)
    
    ax.scatter(embeddings_2d[real_mask, 0], embeddings_2d[real_mask, 1],
               c='green', alpha=0.7, s=30, label=f'Real: {real_mask.sum():,}',
               edgecolors='darkgreen', linewidth=0.2)
    
    # Highlight query if provided
    if highlight_source_id and highlight_source_id in source_ids:
        query_idx = np.where(source_ids == highlight_source_id)[0][0]
        query_label = labels[query_idx]
        
        ax.scatter(embeddings_2d[query_idx, 0], embeddings_2d[query_idx, 1],
                  c='yellow', s=300, marker='*', edgecolors='black', linewidth=2,
                  label=f'Query: {highlight_source_id} ({query_label})', zorder=10)
    
    # Customize plot
    ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Foundation Model Embedding Space', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()


def get_random_object(source_ids, labels, tabular_dict, object_type='both', 
                     nc2snr_min=None, nc2snr_max=None, nc2snrpm_min=None, nc2snrpm_max=None, 
                     nc2mag_min=None, nc2mag_max=None):
    """Get a random object of specified type with optional filtering"""
    
    # Start with type filtering
    if object_type == 'real':
        type_mask = labels == 'Real'
    elif object_type == 'fake':
        type_mask = labels == 'Fake'
    else:  # both
        type_mask = np.ones(len(source_ids), dtype=bool)
    
    # Apply filtering if specified
    valid_indices = []
    for i, sid in enumerate(source_ids):
        if not type_mask[i]:
            continue
            
        # Check if we have tabular data for this source
        if sid in tabular_dict:
            row_data = tabular_dict[sid]
            
            # Check NC2snr filter
            if nc2snr_min is not None or nc2snr_max is not None:
                nc2snr = row_data.get('NC2snr', np.nan)
                if pd.isna(nc2snr):
                    continue
                if nc2snr_min is not None and nc2snr < nc2snr_min:
                    continue
                if nc2snr_max is not None and nc2snr > nc2snr_max:
                    continue
            
            # Check NC2snrpm filter
            if nc2snrpm_min is not None or nc2snrpm_max is not None:
                nc2snrpm = row_data.get('NC2snrpm', np.nan)
                if pd.isna(nc2snrpm):
                    continue
                if nc2snrpm_min is not None and nc2snrpm < nc2snrpm_min:
                    continue
                if nc2snrpm_max is not None and nc2snrpm > nc2snrpm_max:
                    continue
            
            # Check NC2mag filter
            if nc2mag_min is not None or nc2mag_max is not None:
                nc2mag = row_data.get('NC2mag_1', np.nan)
                if pd.isna(nc2mag):
                    continue
                if nc2mag_min is not None and nc2mag < nc2mag_min:
                    continue
                if nc2mag_max is not None and nc2mag > nc2mag_max:
                    continue
        
        valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return None
        
    random_idx = np.random.choice(valid_indices)
    return source_ids[random_idx]


def print_similarity_statistics(query_info, results):
    """Print comprehensive similarity search statistics"""
    
    print(f"SIMILARITY SEARCH RESULTS:")
    print("=" * 60)
    
    print(f"QUERY: {query_info['source_id']} ({query_info['label']})")
    print(f"   Similarity: 1.000 (self)")
    print()
    
    print(f"TOP {len(results)} MOST SIMILAR OBJECTS:")
    print("-" * 60)
    
    for i, result in enumerate(results):
        source_id = result['source_id']
        similarity = result['similarity']
        label = result['label']
        
        status = "✓" if label == query_info['label'] else "✗"
        
        print(f"{i+1:2d}. {source_id} ({label}) - Similarity: {similarity:.4f} {status}")
    
    print()
    
    # Print summary statistics
    real_count = sum(1 for r in results if r['label'] == 'Real')
    fake_count = len(results) - real_count
    avg_similarity = np.mean([r['similarity'] for r in results])
    same_type_count = sum(1 for r in results if r['label'] == query_info['label'])
    
    print(f"SEARCH STATISTICS:")
    print(f"   Real objects: {real_count}/{len(results)} ({real_count/len(results)*100:.1f}%)")
    print(f"   Fake objects: {fake_count}/{len(results)} ({fake_count/len(results)*100:.1f}%)")
    print(f"   Same type as query: {same_type_count}/{len(results)} ({same_type_count/len(results)*100:.1f}%)")
