"""
Meta-Learning Training Engine for Prototypical Few-Shot Direction Detection.

- K is drawn per episode from TRAIN_K_RANGE, so the model is trained for every K it is tested on.
- Early stopping on validation loss, measured on held-out *recordings* (never on random windows).
- Data augmentation: amplitude scaling, additive noise, channel dropout, small time shifts.
"""

import argparse
import copy
import glob
import os

import numpy as np
import torch
import torch.optim as optim

from src.config import (
    TENSORS_DIR, CHECKPOINTS_DIR, N_WAY, K_SHOT, TRAIN_K_RANGE, Q_QUERY,
    LATENT_DIM, TEMPERATURE, LEARNING_RATE, WEIGHT_DECAY, BATCH_EPISODES, NUM_EPOCHS,
    PATIENCE, DROPOUT_RATE, USE_ANGULAR_LOSS, ANGULAR_LAMBDA, SEED, NUM_CHANNELS,
    set_seed, ensure_dirs
)
from src.dataset import EEGSubjectDataset
from src.losses import PrototypicalLoss, DirectionalAngularMarginLoss, compute_prototypes
from src.models import build_model, count_parameters


def augment(x, rng):
    """x: [B, C, T] tensor. Random amplitude scaling, noise, channel dropout and time shift."""
    B, C, T = x.shape
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    scale = 1.0 + 0.1 * torch.randn(B, C, 1, generator=gen)
    x = x * scale + 0.1 * torch.randn(B, C, T, generator=gen)
    keep = (torch.rand(B, C, 1, generator=gen) > 0.1).float()
    x = x * keep
    shift = int(rng.integers(-32, 33))
    return torch.roll(x, shifts=shift, dims=-1)


def split_validation_recordings(dataset, rng):
    """Holds out one recording per subject (subjects with >= 2 recordings) for validation."""
    val_recs = []
    for s in np.unique(dataset.subject):
        recs = np.unique(dataset.rec[dataset.subject == s])
        if len(recs) >= 2:
            val_recs.append(rng.choice(recs))
    val_mask = np.isin(dataset.rec, val_recs)
    return dataset.subset(~val_mask), dataset.subset(val_mask)


def episode_loss(model, batch, criterion, angular, n_way, device):
    s_x, s_y, q_x, q_y = [t.to(device) for t in batch]
    z = model(torch.cat([s_x, q_x]))         # one pass, so BatchNorm sees support and query together
    z_s, z_q = z[:len(s_x)], z[len(s_x):]
    prototypes = compute_prototypes(z_s, s_y, n_way)
    loss, acc, _ = criterion(z_q, prototypes, q_y)
    if angular is not None:
        loss = loss + angular(prototypes)
    return loss, acc


def train_meta_model(
    train_dataset,
    val_dataset=None,
    model_type="spatial_temporal",
    n_way=N_WAY,
    k_range=TRAIN_K_RANGE,
    q_query=Q_QUERY,
    epochs=NUM_EPOCHS,
    episodes_per_epoch=BATCH_EPISODES,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    use_angular=USE_ANGULAR_LOSS,
    use_augmentation=True,
    seed=SEED,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path=None,
    verbose=True,
):
    """
    Trains on `train_dataset`. If `val_dataset` is None, one recording per subject is held out from
    `train_dataset` for early stopping. Returns (model, history).
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    if val_dataset is None:
        train_dataset, val_dataset = split_validation_recordings(train_dataset, rng)

    model = build_model(model_type, NUM_CHANNELS, LATENT_DIM, DROPOUT_RATE).to(device)
    if verbose:
        print(f"--> {model_type}: {count_parameters(model):,} parameters | device {device} | "
              f"train {len(train_dataset)} windows, val {len(val_dataset)} windows | K in {k_range}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = PrototypicalLoss(temperature=TEMPERATURE)
    angular = DirectionalAngularMarginLoss(ANGULAR_LAMBDA) if use_angular else None

    # Fixed validation episodes so the early-stopping signal is not resampling noise
    val_rng = np.random.default_rng(seed + 1)
    val_episodes = []
    if len(val_dataset):
        for _ in range(30):
            try:
                val_episodes.append(val_dataset.sample_fewshot_episode(n_way, int(val_rng.integers(k_range[0], k_range[1] + 1)), q_query, val_rng))
            except ValueError:
                break

    history = []
    best = {'val_loss': np.inf, 'epoch': 0, 'state': copy.deepcopy(model.state_dict())}
    for epoch in range(1, epochs + 1):
        model.train()
        losses, accs = [], []
        for _ in range(episodes_per_epoch):
            k = int(rng.integers(k_range[0], k_range[1] + 1))
            s_x, s_y, q_x, q_y = train_dataset.sample_fewshot_episode(n_way, k, q_query, rng)
            if use_augmentation:
                s_x, q_x = augment(s_x, rng), augment(q_x, rng)
            optimizer.zero_grad()
            loss, acc = episode_loss(model, (s_x, s_y, q_x, q_y), criterion, angular, n_way, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(loss.item())
            accs.append(acc.item())
        scheduler.step()

        val_loss, val_acc = np.nan, np.nan
        if val_episodes:
            model.eval()
            with torch.no_grad():
                vl = [episode_loss(model, ep, criterion, angular, n_way, device) for ep in val_episodes]
            val_loss = float(np.mean([l.item() for l, _ in vl]))
            val_acc = float(np.mean([a.item() for _, a in vl]))
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'train_acc': float(np.mean(accs)),
                        'val_loss': val_loss, 'val_acc': val_acc})
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss {np.mean(losses):.4f} Acc {np.mean(accs) * 100:5.1f}% "
                  f"| Val Loss {val_loss:.4f} Acc {val_acc * 100:5.1f}%", flush=True)

        if not val_episodes:
            best = {'val_loss': np.nan, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict())}
        elif val_loss < best['val_loss']:
            best = {'val_loss': val_loss, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict())}
        elif epoch - best['epoch'] >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch {best['epoch']}, val loss {best['val_loss']:.4f})", flush=True)
            break

    model.load_state_dict(best['state'])
    model.eval()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': best['epoch'],
            'model_state_dict': model.state_dict(),
            'val_loss': best['val_loss'],
            'history': history,
            'config': {'model_type': model_type, 'latent_dim': LATENT_DIM, 'dropout_rate': DROPOUT_RATE,
                       'num_channels': NUM_CHANNELS, 'temperature': TEMPERATURE, 'n_way': n_way,
                       'k_range': list(k_range), 'use_angular': use_angular, 'seed': seed},
        }, save_path)
        if verbose:
            print(f"Saved checkpoint: {save_path}", flush=True)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Few-Shot EEG Direction Model on all subjects")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--model", type=str, default="spatial_temporal", choices=["spatial_temporal", "eegnet"])
    parser.add_argument("--angular", action="store_true", help="Add the directional angular loss")
    parser.add_argument("--alignment", default="euclidean", choices=["none", "euclidean"])
    args = parser.parse_args()

    ensure_dirs()
    npz_files = sorted(glob.glob(os.path.join(TENSORS_DIR, "subject_*.npz")))
    if not npz_files:
        raise SystemExit("[Error] No subject_*.npz files in processed_data/tensors. Run `python -m src.convert_raw_data` first.")
    dataset = EEGSubjectDataset(npz_files, alignment=args.alignment)
    save_ckpt = os.path.join(CHECKPOINTS_DIR, f"protonet_{args.model}.pth")
    train_meta_model(dataset, model_type=args.model, epochs=args.epochs, use_angular=args.angular, save_path=save_ckpt)
