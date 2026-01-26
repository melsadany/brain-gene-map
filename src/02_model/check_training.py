# check_training.py
import torch
import glob

checkpoints = sorted(glob.glob("checkpoints/checkpoint_epoch_*.pt"))

epochs = []
train_mse = []
val_mse = []
val_corr = []

for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, weights_only=False)
    epochs.append(ckpt['epoch'])
    val_mse.append(ckpt['val_metrics']['mse'])
    val_corr.append(ckpt['val_metrics']['mean_gene_corr'])

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, val_mse, marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation MSE')
ax1.set_title('Validation Loss Over Training')
ax1.grid(True)

ax2.plot(epochs, val_corr, marker='o', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean Gene Correlation')
ax2.set_title('Validation Correlation Over Training')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("Saved: training_curves.png")
print(f"\nBest correlation: {max(val_corr):.4f} at epoch {epochs[val_corr.index(max(val_corr))]}")
print(f"Final correlation: {val_corr[-1]:.4f}")
