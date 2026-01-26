# analyze_results.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from model import HierarchicalGeneINR
from dataset import create_dataloaders

config = Config()
train_loader, val_loader, num_genes, gene_names, _ = create_dataloaders(config)

# Load best model
checkpoint = torch.load("checkpoints/best_model.pt", weights_only=False)
model = HierarchicalGeneINR(config, num_genes).to(config.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get predictions
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(config.device)
        preds = model(inputs).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

all_preds = np.concatenate(all_preds, axis=0)  # (N, G)
all_targets = np.concatenate(all_targets, axis=0)

# Per-gene correlations
gene_corrs = []
for g in range(all_preds.shape[1]):
    r = np.corrcoef(all_preds[:, g], all_targets[:, g])[0, 1]
    gene_corrs.append(r if not np.isnan(r) else 0)

gene_corrs = np.array(gene_corrs)
gene_vars = np.var(all_targets, axis=0)

# Analysis
print(f"Correlation distribution:")
print(f"  Mean: {np.mean(gene_corrs):.4f}")
print(f"  Median: {np.median(gene_corrs):.4f}")
print(f"  25th percentile: {np.percentile(gene_corrs, 25):.4f}")
print(f"  75th percentile: {np.percentile(gene_corrs, 75):.4f}")
print(f"  Genes with r>0.5: {np.sum(gene_corrs > 0.5)} ({100*np.sum(gene_corrs > 0.5)/len(gene_corrs):.1f}%)")
print(f"  Genes with r>0.3: {np.sum(gene_corrs > 0.3)} ({100*np.sum(gene_corrs > 0.3)/len(gene_corrs):.1f}%)")
print(f"  Genes with r<0.1: {np.sum(gene_corrs < 0.1)} ({100*np.sum(gene_corrs < 0.1)/len(gene_corrs):.1f}%)")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(gene_corrs, bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(gene_corrs), color='r', linestyle='--', linewidth=2, label=f'Mean={np.mean(gene_corrs):.3f}')
axes[0].axvline(np.median(gene_corrs), color='g', linestyle='--', linewidth=2, label=f'Median={np.median(gene_corrs):.3f}')
axes[0].set_xlabel('Correlation (r)')
axes[0].set_ylabel('Number of genes')
axes[0].set_title('Distribution of per-gene correlations')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(gene_vars, gene_corrs, alpha=0.3, s=5)
axes[1].set_xlabel('Gene variance (in validation set)')
axes[1].set_ylabel('Correlation (r)')
axes[1].set_title('Variance vs. prediction quality')
axes[1].grid(alpha=0.3)

# Top and bottom genes
top_50_idx = np.argsort(gene_corrs)[-50:]
axes[2].barh(range(50), gene_corrs[top_50_idx], color='green', alpha=0.7)
axes[2].set_xlabel('Correlation')
axes[2].set_ylabel('Gene rank')
axes[2].set_title('Top 50 genes by correlation')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gene_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: gene_analysis.png")

# Print top 20 genes
print("\nTop 20 genes:")
for idx in np.argsort(gene_corrs)[-20:][::-1]:
    print(f"  {gene_names[idx]}: r={gene_corrs[idx]:.3f}, var={gene_vars[idx]:.3f}")

print("\nBottom 20 genes:")
for idx in np.argsort(gene_corrs)[:20]:
    print(f"  {gene_names[idx]}: r={gene_corrs[idx]:.3f}, var={gene_vars[idx]:.3f}")
