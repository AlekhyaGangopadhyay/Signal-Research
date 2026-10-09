"""
Meta-Learning Training Engine for Prototypical Few-Shot Direction Detection.
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import (
    TENSORS_DIR, CHECKPOINTS_DIR, N_WAY, K_SHOT, Q_QUERY,
    LATENT_DIM, TEMPERATURE, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_EPISODES, NUM_EPOCHS, DROPOUT_RATE
)
from src.dataset import EEGSubjectDataset
from src.models import SpatialTemporalProtoNet, EEGNetBackbone
from src.losses import PrototypicalLoss, DirectionalAngularMarginLoss

def train_meta_model(
    train_dataset,
    val_dataset=None,
    model_type="spatial_temporal",
    n_way=N_WAY,
    k_shot=K_SHOT,
    q_query=Q_QUERY,
    epochs=NUM_EPOCHS,
    episodes_per_epoch=BATCH_EPISODES,
    lr=LEARNING_RATE,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path=None
):
    print(f"--> Initializing Few-Shot Meta-Learner on Device: {device.upper()}")
    print(f"--> Architecture: {model_type} | N-Way: {n_way} | K-Shot: {k_shot} | Q-Query: {q_query}")

    # Instantiate Model
    if model_type == "spatial_temporal":
        model = SpatialTemporalProtoNet(num_channels=18, latent_dim=LATENT_DIM, dropout_rate=DROPOUT_RATE)
    else:
        model = EEGNetBackbone(num_channels=18, latent_dim=LATENT_DIM, dropout_rate=DROPOUT_RATE)

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    proto_criterion = PrototypicalLoss(temperature=TEMPERATURE)
    angular_criterion = DirectionalAngularMarginLoss(lambda_angular=0.1)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_accs = []

        for _ in range(episodes_per_epoch):
            # 1. Sample episode
            support_x, support_y, query_x, query_y = train_dataset.sample_fewshot_episode(
                n_way=n_way, k_shot=k_shot, q_query=q_query
            )

            support_x = support_x.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            optimizer.zero_grad()

            # 2. Extract latent embeddings
            z_support = model(support_x) # [N*K, D]
            z_query = model(query_x)     # [N*Q, D]

            # 3. Compute class prototypes
            z_support_reshaped = z_support.view(n_way, k_shot, -1)
            prototypes = z_support_reshaped.mean(dim=1) # [N_way, D]

            # 4. Prototypical cross-entropy loss + Angular Margin Regularizer
            loss_proto, acc, _ = proto_criterion(z_query, prototypes, query_y)
            loss_angular = angular_criterion(prototypes)
            total_loss = loss_proto + loss_angular

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_losses.append(total_loss.item())
            train_accs.append(acc.item())

        scheduler.step()

        # Validation phase if val_dataset provided
        val_acc = 0.0
        if val_dataset is not None and len(val_dataset) > 0:
            model.eval()
            val_accs = []
            with torch.no_grad():
                for _ in range(30):
                    s_x, s_y, q_x, q_y = val_dataset.sample_fewshot_episode(n_way, k_shot, q_query)
                    s_x, q_x, q_y = s_x.to(device), q_x.to(device), q_y.to(device)

                    z_s = model(s_x).view(n_way, k_shot, -1).mean(dim=1)
                    z_q = model(q_x)
                    _, acc, _ = proto_criterion(z_q, z_s, q_y)
                    val_accs.append(acc.item())
            val_acc = np.mean(val_accs) * 100

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs) * 100

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc and save_path:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': {'latent_dim': LATENT_DIM, 'n_way': n_way, 'k_shot': k_shot}
            }, save_path)

    if save_path and not os.path.exists(save_path):
        # Save final model if no val dataset
        torch.save({'model_state_dict': model.state_dict()}, save_path)

    print(f"Meta-Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%\n")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Few-Shot EEG Direction Model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--k_shot", type=int, default=K_SHOT)
    parser.add_argument("--model", type=str, default="spatial_temporal", choices=["spatial_temporal", "eegnet"])
    args = parser.parse_args()

    # Load all available subject NPZs
    npz_files = glob.glob(os.path.join(TENSORS_DIR, "*.npz"))
    if not npz_files:
        print("[Error] No .npz files found in processed_data/tensors. Please run convert_raw_data.py first!")
    else:
        dataset = EEGSubjectDataset(npz_files)
        save_ckpt = os.path.join(CHECKPOINTS_DIR, f"protonet_{args.model}_{args.k_shot}shot.pth")
        train_meta_model(dataset, model_type=args.model, k_shot=args.k_shot, epochs=args.epochs, save_path=save_ckpt)
