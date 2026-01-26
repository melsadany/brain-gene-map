"""PyTorch dataset for AHBA gene expression"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AHBAGeneDataset(Dataset):
    def __init__(self, csv_path, donor_filter=None, structure_filter=None):
        """
        Args:
            csv_path: Path to normalized CSV
            donor_filter: List of donor IDs to include (None = all)
            structure_filter: List of structures to include (None = all)
        """
        df = pd.read_csv(csv_path)
        
        # Apply filters
        if donor_filter is not None:
            df = df[df['donor_id'].isin(donor_filter)]
        if structure_filter is not None:
            df = df[df['structure'].isin(structure_filter)]
        
        # Identify columns
        self.meta_cols = ['mni_x', 'mni_y', 'mni_z', 'donor_id', 'structure']
        self.gene_cols = [c for c in df.columns if c not in self.meta_cols]
        
        # Structure encoding
        self.structure_map = {'CX': 0, 'CB': 1, 'BS': 2}
        df['structure_idx'] = df['structure'].map(self.structure_map)
        
        # Normalize coordinates to [-1, 1]
        self.mni_bounds = {
            'mni_x': (-90, 90),
            'mni_y': (-126, 90),
            'mni_z': (-72, 108)
        }
        
        coords_norm = []
        for coord in ['mni_x', 'mni_y', 'mni_z']:
            lo, hi = self.mni_bounds[coord]
            normalized = 2 * (df[coord].values - lo) / (hi - lo) - 1
            coords_norm.append(normalized)
        
        self.coords = np.stack(coords_norm, axis=1).astype(np.float32)  # (N, 3)
        self.structure_idx = df['structure_idx'].values.astype(np.int64)  # (N,)
        self.expression = df[self.gene_cols].values.astype(np.float32)  # (N, G)
        
        # Store metadata for later use
        self.donor_ids = df['donor_id'].values
        self.structures = df['structure'].values
        
        print(f"Loaded dataset: {len(self)} samples, {len(self.gene_cols)} genes")
        print(f"  Structures: {dict(zip(*np.unique(self.structure_idx, return_counts=True)))}")
    
    def __len__(self):
        return self.coords.shape[0]
    
    def __getitem__(self, idx):
        """
        Returns:
            inputs: (x, y, z, structure_idx) - shape (4,)
            targets: gene expression vector - shape (n_genes,)
        """
        coord = self.coords[idx]  # (3,)
        struct = self.structure_idx[idx]  # scalar
        expr = self.expression[idx]  # (n_genes,)
        
        # Combine into input vector
        inputs = np.concatenate([coord, [struct]])  # (4,)
        
        # IMPORTANT: ensure float32
        inputs = inputs.astype(np.float32) 
        
        return torch.from_numpy(inputs), torch.from_numpy(expr)
    
    def get_num_genes(self):
        return len(self.gene_cols)
    
    def get_gene_names(self):
        return self.gene_cols


def create_dataloaders(config, top_n_genes=None):
    """
    Create train and validation dataloaders
    
    Args:
        top_n_genes: If specified, only use top N genes by variance
    """
    from torch.utils.data import DataLoader
    
    if config.val_strategy == "donor":
        all_donors_df = pd.read_csv(config.data_path)
        all_donors = all_donors_df['donor_id'].unique().tolist()
        train_donors = [d for d in all_donors if d not in config.val_donors]
        
        train_dataset = AHBAGeneDataset(config.data_path, donor_filter=train_donors)
        val_dataset = AHBAGeneDataset(config.data_path, donor_filter=config.val_donors)
    
    elif config.val_strategy == "spatial":
        from sklearn.model_selection import GroupKFold
        df = pd.read_csv(config.data_path)
        
        df['spatial_group'] = (
            (df['mni_x'] // 20).astype(str) + '_' +
            (df['mni_y'] // 20).astype(str) + '_' +
            (df['mni_z'] // 20).astype(str)
        )
        
        gkf = GroupKFold(n_splits=5)
        train_idx, val_idx = next(gkf.split(df, groups=df['spatial_group']))
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # DROP spatial_group before saving
        train_df = train_df.drop('spatial_group', axis=1) 
        val_df = val_df.drop('spatial_group', axis=1)
        
        train_df.to_csv('train_temp.csv', index=False)
        val_df.to_csv('val_temp.csv', index=False)
        
        train_dataset = AHBAGeneDataset('train_temp.csv')
        val_dataset = AHBAGeneDataset('val_temp.csv')
    
    # GENE FILTERING: Select top N genes by variance
    if top_n_genes is not None:
        gene_vars = np.var(train_dataset.expression, axis=0)
        top_gene_idx = np.argsort(gene_vars)[-top_n_genes:]
        
        # Filter both datasets
        train_dataset.expression = train_dataset.expression[:, top_gene_idx]
        val_dataset.expression = val_dataset.expression[:, top_gene_idx]
        
        # Update gene names
        train_dataset.gene_cols = [train_dataset.gene_cols[i] for i in top_gene_idx]
        val_dataset.gene_cols = train_dataset.gene_cols
        
        print(f"  Filtered to top {top_n_genes} genes by variance")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.get_num_genes(), train_dataset.get_gene_names(), train_dataset

