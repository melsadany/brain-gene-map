"""Generate full-brain gene expression predictions"""

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import time

from config import Config
from model import HierarchicalGeneINR


def create_mni_grid(resolution_mm, bounds):
    """Create 3D grid of MNI coordinates"""
    x = np.arange(bounds['x'][0], bounds['x'][1] + resolution_mm, resolution_mm)
    y = np.arange(bounds['y'][0], bounds['y'][1] + resolution_mm, resolution_mm)
    z = np.arange(bounds['z'][0], bounds['z'][1] + resolution_mm, resolution_mm)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    shape = (len(x), len(y), len(z))
    
    print(f"Created {resolution_mm}mm MNI grid:")
    print(f"  X: {bounds['x'][0]} to {bounds['x'][1]} mm ({len(x)} voxels)")
    print(f"  Y: {bounds['y'][0]} to {bounds['y'][1]} mm ({len(y)} voxels)")
    print(f"  Z: {bounds['z'][0]} to {bounds['z'][1]} mm ({len(z)} voxels)")
    print(f"  Total: {coords.shape[0]:,} voxels")
    print(f"  Shape: {shape}")
    
    return coords, shape


def normalize_coords(coords, bounds):
    """Normalize MNI coordinates to [-1, 1]"""
    coords_norm = np.zeros_like(coords)
    
    for i, axis in enumerate(['x', 'y', 'z']):
        lo, hi = bounds[axis]
        coords_norm[:, i] = 2 * (coords[:, i] - lo) / (hi - lo) - 1
    
    return coords_norm.astype(np.float32)


def assign_structure_labels_from_fsl_atlas(coords, atlas_path):
    """
    Assign CTX / CRB / SCTX using FSL labeled atlas
    """
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    inv_affine = np.linalg.inv(affine)

    N = coords.shape[0]
    coords_h = np.c_[coords, np.ones(N)]
    vox = coords_h @ inv_affine.T
    vox = np.round(vox[:, :3]).astype(int)

    cerebellum_labels = {6, 7, 8, 45, 46, 47}
    cortex_labels = {3, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 
                     42, 55, 56, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 83, 84}
    ignore_labels = {0, 24}

    structure_idx = np.zeros(N, dtype=np.int64)

    for i, (x, y, z) in enumerate(vox):
        if (0 <= x < atlas_data.shape[0] and
            0 <= y < atlas_data.shape[1] and
            0 <= z < atlas_data.shape[2]):
            label = int(atlas_data[x, y, z])
            if label in cerebellum_labels:
                structure_idx[i] = 1
            elif label in cortex_labels:
                structure_idx[i] = 0
            else:
                structure_idx[i] = 2
        else:
            structure_idx[i] = 2

    unique, counts = np.unique(structure_idx, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {['CTX','CRB','SCTX'][u]}: {c:,} voxels ({100*c/N:.1f}%)")

    return structure_idx


def predict_brain_batch(model, coords_norm, structure_idx, config, batch_size=50000):
    """Predict gene expression for all brain coordinates"""
    model.eval()
    N = coords_norm.shape[0]
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="Predicting"):
            batch_end = min(i + batch_size, N)
            
            coords_batch = torch.from_numpy(coords_norm[i:batch_end]).to(config.device)
            struct_batch = torch.from_numpy(structure_idx[i:batch_end]).float().to(config.device)
            
            inputs = torch.cat([coords_batch, struct_batch.unsqueeze(1)], dim=1)
            pred = model(inputs).detach().cpu().numpy()
            predictions.append(pred)
    
    predictions = np.concatenate(predictions, axis=0)
    print(f"Generated predictions: {predictions.shape}")
    return predictions


def save_as_nifti(data_3d, resolution_mm, output_path, gene_name=None, atlas_affine=None):
    """Save 3D array as NIfTI file with proper affine"""
    if atlas_affine is not None:
        affine = atlas_affine.copy()
    else:
        affine = np.array([
            [resolution_mm, 0, 0, -90],
            [0, resolution_mm, 0, -126],
            [0, 0, resolution_mm, -72],
            [0, 0, 0, 1]
        ])
    
    nii = nib.Nifti1Image(data_3d.astype(np.float32), affine)
    
    if gene_name:
        nii.header['descrip'] = f'Predicted expression: {gene_name}'.encode()
    
    nib.save(nii, output_path)


def save_predictions(predictions, coords, shape, gene_names, resolution_mm, output_dir, atlas_affine=None):
    """Save predictions in multiple formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSaving predictions to {output_dir}/")
    
    # 1. Save as compressed numpy array
    print("  Saving numpy array...")
    np.savez_compressed(
        output_dir / f'predictions_{resolution_mm}mm.npz',
        predictions=predictions.astype(np.float32),
        coords=coords.astype(np.float32),
        gene_names=np.array(gene_names, dtype=object),
        shape=np.array(shape)
    )
    
    # 2. Save metadata CSV
    print("  Saving metadata...")
    metadata = pd.DataFrame({
        'gene_name': gene_names,
        'mean_expression': predictions.mean(axis=0),
        'std_expression': predictions.std(axis=0),
        'min_expression': predictions.min(axis=0),
        'max_expression': predictions.max(axis=0)
    })
    metadata.to_csv(output_dir / f'gene_metadata_{resolution_mm}mm.csv', index=False)
    
    # 3. Save top 100 genes as NIfTI
    print("  Saving top 100 genes as NIfTI files...")
    nifti_dir = output_dir / f'nifti_{resolution_mm}mm'
    nifti_dir.mkdir(exist_ok=True)
    
    gene_vars = predictions.var(axis=0)
    top_gene_idx = np.argsort(gene_vars)[-100:][::-1]
    
    for rank, gene_idx in enumerate(tqdm(top_gene_idx, desc="  Saving NIfTI")):
        gene_name = gene_names[gene_idx]
        data_3d = predictions[:, gene_idx].reshape(shape)
        
        safe_gene_name = gene_name.replace('/', '_').replace('\\', '_')
        output_path = nifti_dir / f'{rank+1:03d}_{safe_gene_name}.nii.gz'
        save_as_nifti(data_3d, resolution_mm, output_path, gene_name, atlas_affine)
    
    print(f"\n✓ Saved all predictions to {output_dir}/")


def main():
    config = Config()
    atlas_path = "/Dedicated/jmichaelson-wdata/msmuhammad/refs/labeled-MNI/2mm/FS-anat/FS-labeled_resampled-2mm.nii.gz"
    
    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load(f"{config.checkpoint_dir}/best_model.pt", 
                           weights_only=False, 
                           map_location=config.device)
    
    num_genes = len(checkpoint['gene_names'])
    gene_names = checkpoint['gene_names']
    
    print(f"  Model trained on {num_genes} genes")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Val correlation: {checkpoint['val_metrics']['mean_gene_corr']:.4f}")
    
    model = HierarchicalGeneINR(config, num_genes).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load atlas for affine
    atlas_img = nib.load(atlas_path)
    atlas_affine = atlas_img.affine
    
    # Generate predictions for both resolutions
    for resolution_mm in [config.resolution_2mm, config.resolution_8mm]:
        print(f"\n{'='*60}")
        print(f"Generating {resolution_mm}mm resolution predictions")
        print(f"{'='*60}\n")
        
        coords, shape = create_mni_grid(resolution_mm, config.mni_bounds)
        structure_idx = assign_structure_labels_from_fsl_atlas(coords, atlas_path)
        coords_norm = normalize_coords(coords, config.mni_bounds)
        
        print("\nPredicting gene expression...")
        start_time = time.time()
        predictions = predict_brain_batch(model, coords_norm, structure_idx, config, batch_size=50000)
        elapsed = time.time() - start_time
        print(f"  Prediction time: {elapsed:.1f}s ({coords.shape[0]/elapsed:.0f} voxels/sec)")
        
        output_dir = f"predictions/{resolution_mm}mm"
        save_predictions(predictions, coords, shape, gene_names, resolution_mm, output_dir, atlas_affine)
    
    print(f"\n{'='*60}")
    print("All predictions complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
