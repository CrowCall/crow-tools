from asteroid.data import LibriMix

# Automatically downloads MiniLibriMix
train_loader, val_loader = LibriMix.loaders_from_mini(
    task="sep_clean",  # "sep_clean" means separating clean speech from mixture
    batch_size=1       # Just to load one file at a time
)

