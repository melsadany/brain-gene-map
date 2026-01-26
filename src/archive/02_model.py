import pandas as pd
import numpy as np
from pathlib import Path

# Path to corrected gex
csv_path = Path("data/derivates/gex-corrected.csv")
df = pd.read_csv(csv_path)

print(df.shape)
print(df.head())
print(df.columns[:10])

meta_cols = ['mni_x', 'mni_y', 'mni_z', 'donor_id']
gene_cols = [c for c in df.columns if c not in meta_cols]

print(f"Number of genes: {len(gene_cols)}")

# Typical MNI bounds for adult brain (can be adjusted)
MNI_BOUNDS = {
    'mni_x': (-90, 90),
    'mni_y': (-126, 90),
    'mni_z': (-72, 108),
}

for coord in ['mni_x', 'mni_y', 'mni_z']:
    lo, hi = MNI_BOUNDS[coord]
    df[f'{coord}_norm'] = 2 * (df[coord] - lo) / (hi - lo) - 1

print(df[[c for c in df.columns if c.endswith('_norm')]].describe())

from scipy.linalg import svd

# Expression matrix: samples × genes
X = df[gene_cols].values.astype(np.float32)
print("Expression matrix:", X.shape)  # (N_samples, N_genes)

# Gene–gene correlation: N_genes × N_genes
print("Computing gene–gene correlation...")
gene_corr = np.corrcoef(X.T)  # correlation across samples
print("gene_corr shape:", gene_corr.shape)

# SVD for spectral embedding
print("Performing SVD...")
U, S, Vt = svd(gene_corr, full_matrices=False)

n_components = 64  # as in INR paper
gene_embedding = U[:, :n_components].astype(np.float32)  # (N_genes, 64)

explained_var = (S[:n_components]**2).sum() / (S**2).sum()
print(f"Explained variance by top {n_components} components: {explained_var:.2%}") #99.95
print("Gene embedding shape:", gene_embedding.shape) #17252,64

# Map gene name -> index
gene_to_idx = {g: i for i, g in enumerate(gene_cols)}

import json
import numpy as np

np.save("gene_embedding_64d.npy", gene_embedding)
with open("gene_to_idx.json", "w") as f:
    json.dump(gene_to_idx, f, indent=2)
    
    
df['anatomical_label'] = 1.0  # all GM; if you later add WM, use -1.0 for WM


## long-format training data
import pyarrow as pa
import pyarrow.parquet as pq

# Load embedding & mapping
gene_embedding = np.load("gene_embedding_64d.npy")
with open("gene_to_idx.json") as f:
    gene_to_idx = json.load(f)

# Columns we need from df for inputs
coord_cols = ['mni_x_norm', 'mni_y_norm', 'mni_z_norm']
meta_cols_extended = ['mni_x', 'mni_y', 'mni_z', 'donor_id', 'anatomical_label'] + coord_cols

# Prepare Parquet writer
output_path = "training_data.parquet"
writer = None

# Process in row-chunks to avoid huge memory blow-up
chunk_size = 1000  # adjust based on your RAM
n_samples = df.shape[0]

for start in range(0, n_samples, chunk_size):
    end = min(start + chunk_size, n_samples)
    df_chunk = df.iloc[start:end]

    # Extract meta and expression
    meta_chunk = df_chunk[meta_cols_extended].reset_index(drop=True)
    expr_chunk = df_chunk[gene_cols].reset_index(drop=True).to_numpy()  # (chunk_size, n_genes)
    
    # Build long-format rows for this chunk
    # We will create arrays directly for speed
    n_chunk = expr_chunk.shape[0]
    n_genes = expr_chunk.shape[1]
    
    # Repeat coordinates & anatomical labels per gene
    x = np.repeat(meta_chunk['mni_x_norm'].to_numpy(), n_genes)
    y = np.repeat(meta_chunk['mni_y_norm'].to_numpy(), n_genes)
    z = np.repeat(meta_chunk['mni_z_norm'].to_numpy(), n_genes)
    anatomical = np.repeat(meta_chunk['anatomical_label'].to_numpy(), n_genes)
    donor = np.repeat(meta_chunk['donor_id'].to_numpy(), n_genes)
    
    # Gene indices 0..n_genes-1 repeated for each sample in chunk
    gene_idx_vec = np.tile(np.arange(n_genes, dtype=np.int32), n_chunk)
    
    # Expression values flattened sample-major
    expr_vec = expr_chunk.reshape(-1).astype(np.float32)
    
    # Gene embeddings: for each long row, pick embedding for that gene
    embed_mat = gene_embedding[gene_idx_vec]  # shape: (n_chunk*n_genes, 64)
    
    # Build PyArrow table
    data_dict = {
        'x_norm': x.astype(np.float32),
        'y_norm': y.astype(np.float32),
        'z_norm': z.astype(np.float32),
        'anatomical_label': anatomical.astype(np.float32),
        'donor_id': donor.astype(str),
        'gene_idx': gene_idx_vec.astype(np.int32),
        'expression': expr_vec,
    }
    
    # add embedding dimensions
    for j in range(embed_mat.shape[1]):
        data_dict[f'embed_{j}'] = embed_mat[:, j]
    
    table = pa.Table.from_pydict(data_dict)
    
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
    writer.write_table(table)
    
    print(f"Processed rows {start}–{end} → {table.num_rows} training examples")

if writer is not None:
    writer.close()
    print(f"Saved long-format training data to {output_path}")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq

class GeneExpressionDataset(Dataset):
    def __init__(self, parquet_path, gene_count=None, donor_filter=None):
        self.parquet_path = parquet_path
        self.table = pq.read_table(parquet_path)
        df = self.table.to_pandas()  # for now; for huge data, use streaming
        
        if donor_filter is not None:
            df = df[df['donor_id'].isin(donor_filter)]
        
        self.x = df[['x_norm', 'y_norm', 'z_norm']].values.astype('float32')
        self.anatomical = df[['anatomical_label']].values.astype('float32')
        embed_cols = [c for c in df.columns if c.startswith('embed_')]
        self.embed = df[embed_cols].values.astype('float32')
        self.y = df['expression'].values.astype('float32').reshape(-1, 1)
        
        # pack inputs
        self.inputs = np.concatenate([self.x, self.anatomical, self.embed], axis=1)
        self.inputs = torch.from_numpy(self.inputs)
        self.targets = torch.from_numpy(self.y)
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Simple SIREN layer
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.0):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.act = Sine(w0=w0)
    def forward(self, x):
        return self.act(self.linear(x))

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden=512, dim_out=1, num_layers=12):
        super().__init__()
        layers = []
        layers.append(SirenLayer(dim_in, dim_hidden, w0=30.0))
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(dim_hidden, dim_hidden, w0=1.0))
        layers.append(nn.Linear(dim_hidden, dim_out))  # last layer linear
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Usage
train_donors = ['H0351.2001', 'H0351.2002', 'H0351.1009', 'H0351.1012', 'H0351.1015']
val_donor = ['H0351.1016']

train_dataset = GeneExpressionDataset("training_data.parquet", donor_filter=train_donors)
val_dataset = GeneExpressionDataset("training_data.parquet", donor_filter=val_donor)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)

model = SirenNet(dim_in=train_dataset.inputs.shape[1], dim_hidden=512, dim_out=1, num_layers=12)
model = model.cuda()  # if GPU available

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb = xb.cuda()
        yb = yb.cuda()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # simple val
    model.eval()
    with torch.no_grad():
        val_losses = []
        for xb, yb in val_loader:
            xb = xb.cuda()
            yb = yb.cuda()
            pred = model(xb)
            val_losses.append(loss_fn(pred, yb).item())
    print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={np.mean(val_losses):.4f}")



