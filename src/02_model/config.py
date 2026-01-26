"""Configuration for brain gene expression INR model"""

import torch

class Config:
    # Data
    data_path = "/Dedicated/jmichaelson-wdata/msmuhammad/projects/brain-gene-map/data/derivatives/gex-corrected-normalized.csv"
    
    # Training
    batch_size = 2048
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Validation split (leave-one-donor-out or spatial)
    val_strategy = "donor"  # "donor" or "spatial"
    val_donors = ["H0351.1016"]  # if using donor strategy
    
    # Model architecture
    spatial_dim = 256        # spatial latent code dimension
    gene_latent_dim = 128     # gene bottleneck dimension
    hidden_dim = 512         # SIREN hidden layer size
    num_layers = 12          # SIREN depth
    w0_first = 30.0          # SIREN frequency for first layer
    w0_hidden = 1.0          # SIREN frequency for hidden layers
    
    # Loss weights
    lambda_smoothness = 0.01  # spatial smoothness regularization
    smoothness_radius = 5.0  # mm for smoothness sampling
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpointing
    checkpoint_dir = "checkpoints"
    save_every = 5  # epochs
    
    # Inference
    mni_bounds = {
        'x': (-90, 90),
        'y': (-126, 90), 
        'z': (-72, 108)
    }
    resolution_2mm = 2.0
    resolution_8mm = 8.0
