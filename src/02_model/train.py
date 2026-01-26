"""Training script for brain gene expression INR"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from config import Config
from model import HierarchicalGeneINR
from dataset import create_dataloaders


def spatial_smoothness_loss(model, coords, structure_idx, radius):
    """
    Regularization: nearby coordinates should have similar expression
    
    Args:
        coords: (batch, 3) normalized coordinates
        structure_idx: (batch,) structure indices
        radius: sampling radius in normalized coordinate space
    """
    batch_size = coords.shape[0]
    
    # Sample nearby coordinates
    noise = torch.randn_like(coords) * (radius / 90.0)  # scale by MNI range
    coords_nearby = coords + noise
    coords_nearby = torch.clamp(coords_nearby, -1, 1)  # stay in valid range
    
    # Predict at original and nearby points
    inputs_orig = torch.cat([coords, structure_idx.unsqueeze(1).float()], dim=1)
    inputs_nearby = torch.cat([coords_nearby, structure_idx.unsqueeze(1).float()], dim=1)
    
    expr_orig = model(inputs_orig)
    expr_nearby = model(inputs_nearby)
    
    # Penalize large changes
    smoothness = torch.mean((expr_orig - expr_nearby) ** 2)
    
    return smoothness


def train_epoch(model, train_loader, optimizer, config, epoch, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_smooth = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for inputs, targets in pbar:
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        
        optimizer.zero_grad()
        
        # USE MIXED PRECISION
        with autocast():
            predictions = model(inputs)
            mse_loss = nn.functional.mse_loss(predictions, targets)
            
            coords = inputs[:, :3]
            structure_idx = inputs[:, 3].long()
            smooth_loss = spatial_smoothness_loss(
                model, coords, structure_idx, config.smoothness_radius
            )
            
            loss = mse_loss + config.lambda_smoothness * smooth_loss
        
        # Backward with scaler
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_smooth += smooth_loss.item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'mse': mse_loss.item(),
            'smooth': smooth_loss.item()
        })
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'smoothness': total_smooth / n_batches
    }


@torch.no_grad()
def validate(model, val_loader, config):
    """Validation"""
    model.eval()
    total_mse = 0
    all_predictions = []
    all_targets = []
    
    for inputs, targets in tqdm(val_loader, desc="Validating"):
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        
        predictions = model(inputs)
        mse = nn.functional.mse_loss(predictions, targets)
        total_mse += mse.item()
        
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    # Compute gene-wise correlations
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, G)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, G)
    
    gene_correlations = []
    for g in range(all_predictions.shape[1]):
        corr = np.corrcoef(all_predictions[:, g], all_targets[:, g])[0, 1]
        if not np.isnan(corr):
            gene_correlations.append(corr)
    
    mean_corr = np.mean(gene_correlations) if gene_correlations else 0
    median_corr = np.median(gene_correlations) if gene_correlations else 0
    
    return {
        'mse': total_mse / len(val_loader),
        'mean_gene_corr': mean_corr,
        'median_gene_corr': median_corr
    }


def main():
    config = Config()
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter('runs/gene_inr')
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, num_genes, gene_names, train_dataset = create_dataloaders(config,top_n_genes=5000)

    
    # Create model
    print("\nInitializing model...")
    model = HierarchicalGeneINR(config, num_genes).to(config.device)
    
    # Optimizer with separate learning rates for gene decoders
    optimizer = torch.optim.AdamW([
        {'params': model.encoder_ctx.parameters(), 'lr': config.learning_rate},
        {'params': model.encoder_crb.parameters(), 'lr': config.learning_rate},
        {'params': model.encoder_sctx.parameters(), 'lr': config.learning_rate},
        {'params': model.gene_decoder_ctx.parameters(), 'lr': config.learning_rate * 3},
        {'params': model.gene_decoder_crb.parameters(), 'lr': config.learning_rate * 3},
        {'params': model.gene_decoder_sctx.parameters(), 'lr': config.learning_rate * 3}
    ], weight_decay=config.weight_decay)
    
    scaler = GradScaler()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_corr = -1
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, epoch, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, config)
        
        # Scheduler step
        scheduler.step(val_metrics['mse'])
        
        # Log metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}")
        print(f"  Val   - MSE: {val_metrics['mse']:.4f}, Mean Gene Corr: {val_metrics['mean_gene_corr']:.4f}")
        
        # Save checkpoint
        if epoch % config.save_every == 0 or val_metrics['mean_gene_corr'] > best_val_corr:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'gene_names': gene_names
            }
            
            # Save regular checkpoint
            torch.save(checkpoint, f"{config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
            
            # Save best model
            if val_metrics['mean_gene_corr'] > best_val_corr:
                best_val_corr = val_metrics['mean_gene_corr']
                torch.save(checkpoint, f"{config.checkpoint_dir}/best_model.pt")
                print(f"  ✓ Saved best model (corr={best_val_corr:.4f})")
    
    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
