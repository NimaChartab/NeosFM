"""
NEOS Dataset Loading Utilities
=============================

Dataset classes for loading astronomical images and tabular data.
Handles IPAC format files, FITS images, and proper data normalization.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
import warnings
from typing import List, Union

warnings.filterwarnings('ignore')


class NEOSDataset(Dataset):
    """Dataset for NEOS astronomical data with image and tabular features"""
    
    def __init__(self, 
                 visit_paths: List[str],
                 max_samples_per_visit=None,
                 selected_features=None):
        
        # Default feature set (50 selected features)
        if selected_features is None:
            self.selected_features = [
                'NC2sigm_1', 'NC2mag_1', 'NC2flg1', 'NC1sgsk', 'PMRA', 'NC2flux', 'NC2flux_pm',
                'sigPMDec', 'PMDec', 'NC2Npix', 'NC2mag_3', 'sigPMRA', 'NC2snr', 'NC1sigflux',
                'fNC2_exp1', 'uNC2_exp2', 'NC2sigm_3', 'uNC2_exp1', 'NC1sky', 'NC2sigflux_pm',
                'NC2sigmpr', 'uNC2_exp3', 'NC1sigflux_pm', 'fNC2_exp2', 'NC1mag_1', 'NC2sigmpr_pm',
                'uNC2_exp5', 'NC1snr', 'NC2sky', 'NC2mag_8', 'NC2Bad', 'uNC2_exp4', 'NC2neg',
                'uNC2_exp6', 'NC2mpr', 'NC2sigm_7', 'NC2snrpm', 'NC1sigm_1', 'NC2sgm', 'fNC2_exp3',
                'NC2sigflux', 'NC1sigm_7', 'NC1mag_7', 'NC1flux', 'NC1flg1', 'uNC1_exp5',
                'uNC1_exp2', 'uNC1_exp1', 'NC2Sat', 'NC2mpr_pm'
            ]
        else:
            self.selected_features = selected_features
        
        self.visit_datasets = []
        self.visit_names = []
        
        # Load datasets from each visit
        for visit_path in visit_paths:
            try:
                visit_name = Path(visit_path).name
                visit_dataset = self._create_single_visit_dataset(
                    visit_path, max_samples_per_visit
                )
                
                if len(visit_dataset) > 0:
                    self.visit_datasets.append(visit_dataset)
                    self.visit_names.append(visit_name)
                    
            except Exception as e:
                print(f"Error loading {visit_path}: {e}")
                continue
        
        if not self.visit_datasets:
            raise ValueError("No valid datasets loaded from any visit!")
        
        # Combine all datasets
        self.combined_dataset = ConcatDataset(self.visit_datasets)
        
    def _create_single_visit_dataset(self, visit_path, max_samples):
        """Create dataset for a single visit"""
        
        class SingleVisitDataset(Dataset):
            def __init__(self, visit_path, selected_features, max_samples):
                self.visit_path = Path(visit_path)
                self.selected_features = selected_features
                
                # Find files
                tbl_files = list(self.visit_path.glob("*mdexfilt_l4b.tbl"))
                nc1_dirs = list(self.visit_path.glob("*nc1_cutouts_l4b"))
                nc2_dirs = list(self.visit_path.glob("*nc2_cutouts_l4b"))
                
                if not tbl_files or not nc1_dirs or not nc2_dirs:
                    raise ValueError(f"Missing files in {visit_path}")
                
                # Load tabular data
                tabular_file = tbl_files[0]
                self.nc1_cutouts_dir = nc1_dirs[0]
                self.nc2_cutouts_dir = nc2_dirs[0]
                
                with open(tabular_file, 'r') as f:
                    content = f.read()
                table = Table.read(content.splitlines(), format='ascii.ipac')
                tabular_data = table.to_pandas()
                
                # Filter for available features
                available_features = [f for f in selected_features if f in tabular_data.columns]
                missing_features = [f for f in selected_features if f not in tabular_data.columns]
                
                self.selected_features = available_features
                
                # Handle missing values with mean imputation
                self.feature_data = tabular_data[['source_id'] + self.selected_features].copy()
                
                # Normalization: Fill NaN with mean, then z-score normalize
                feature_columns = self.feature_data[self.selected_features]
                feature_columns_filled = feature_columns.fillna(feature_columns.mean())
                
                # Z-score normalization (mean=0, std=1)
                self.feature_mean = feature_columns_filled.mean(axis=0)
                self.feature_std = feature_columns_filled.std(axis=0) + 1e-8
                
                self.feature_data[self.selected_features] = (feature_columns_filled - self.feature_mean) / self.feature_std
                
                # Limit samples if specified
                if max_samples is not None:
                    self.feature_data = self.feature_data.head(max_samples)
                
            def apply_z_scale(self, image, percentile_low=1, percentile_high=99):
                """Apply z-scale normalization to images"""
                valid_pixels = image[~np.isnan(image)]
                if len(valid_pixels) == 0:
                    return np.zeros_like(image)
                    
                p_low, p_high = np.percentile(valid_pixels, [percentile_low, percentile_high])
                
                if p_low == p_high:
                    min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
                    if min_val == max_val:
                        return np.zeros_like(image)
                    return np.clip((image - min_val) / (max_val - min_val), 0, 1)
                
                scaled_image = np.clip((image - p_low) / (p_high - p_low), 0, 1)
                return scaled_image
                
            def __len__(self):
                return len(self.feature_data)
            
            def __getitem__(self, idx):
                row = self.feature_data.iloc[idx]
                source_id = row['source_id']
                
                try:
                    # Load NC1 image
                    nc1_file = self.nc1_cutouts_dir / f"{source_id}_nc1_L4b.fits"
                    with fits.open(nc1_file) as hdul:
                        nc1_data = hdul[0].data.astype(np.float32)
                        nc1_scaled = self.apply_z_scale(nc1_data)
                    
                    # Load NC2 image  
                    nc2_file = self.nc2_cutouts_dir / f"{source_id}_nc2_L4b.fits"
                    with fits.open(nc2_file) as hdul:
                        nc2_data = hdul[0].data.astype(np.float32)
                        nc2_scaled = self.apply_z_scale(nc2_data)
                    
                    # Convert to tensors and add channel dimension
                    nc1_tensor = torch.from_numpy(nc1_scaled).float().unsqueeze(0)
                    nc2_tensor = torch.from_numpy(nc2_scaled).float().unsqueeze(0)
                    
                    # Get normalized tabular features  
                    features = row[self.selected_features].values.astype(np.float32)
                    tabular_tensor = torch.from_numpy(features).float()
                    
                    return {
                        'source_id': source_id,
                        'nc1_image': nc1_tensor,
                        'nc2_image': nc2_tensor, 
                        'tabular_data': tabular_tensor
                    }
                    
                except Exception as e:
                    # Return dummy data on error
                    return {
                        'source_id': source_id,
                        'nc1_image': torch.zeros(1, 61, 61),
                        'nc2_image': torch.zeros(1, 61, 61),
                        'tabular_data': torch.zeros(len(self.selected_features))
                    }
        
        return SingleVisitDataset(visit_path, self.selected_features, max_samples)
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]


def create_dataloader(visit_paths: List[str],
                     batch_size=32,
                     max_samples_per_visit=None,
                     num_workers=2,
                     shuffle=True):
    """Create a DataLoader for NEOS data"""
    
    dataset = NEOSDataset(
        visit_paths=visit_paths,
        max_samples_per_visit=max_samples_per_visit
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return dataset, dataloader


def load_tabular_metadata(visit_paths: List[str]):
    """Load tabular metadata from all visits for label extraction"""
    
    tabular_dict = {}
    
    for visit_path in visit_paths:
        visit_path = Path(visit_path)
        tbl_files = list(visit_path.glob("*mdexfilt_l4b.tbl"))
        
        if tbl_files:
            tbl_file = tbl_files[0]
            
            with open(tbl_file, 'r') as f:
                content = f.read()
            table = Table.read(content.splitlines(), format='ascii.ipac')
            tabular_data = table.to_pandas()
            
            # Add to combined tabular dict
            visit_dict = tabular_data.set_index('source_id').to_dict('index')
            tabular_dict.update(visit_dict)
    
    return tabular_dict


def extract_labels(source_ids, tabular_dict):
    """Extract Real/Fake labels from knownssoname column"""
    
    labels = []
    
    for source_id in source_ids:
        if source_id in tabular_dict:
            knownsso = tabular_dict[source_id].get('knownssoname', np.nan)
            
            if pd.notna(knownsso) and str(knownsso).strip() != '' and str(knownsso) != 'nan':
                labels.append('Real')
            else:
                labels.append('Fake')
        else:
            labels.append('Fake')
    
    return np.array(labels)
