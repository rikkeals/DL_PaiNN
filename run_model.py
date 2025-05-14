"""
Train the PaiNN model on QM9 dataset.
This script prepares the data, creates the model, trains it, evaluates it, and saves the results.
saves the model, training summary, and per-molecule errors.
"""
import torch
import argparse
import time
import json
import os
import torch.nn.functional as F
from tqdm import trange
from src.data import QM9DataModule
from pytorch_lightning import seed_everything
from src.models import PaiNN, AtomwisePostProcessing


def cli():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--target', default=7, type=int)
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_inference', default=1000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--splits', nargs=3, default=[110000, 10000, 10831], type=int)
    parser.add_argument('--subset_size', default=None, type=int)

    # Model
    parser.add_argument('--num_message_passing_layers', default=3, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--num_rbf_features', default=20, type=int)
    parser.add_argument('--num_unique_atoms', default=100, type=int)
    parser.add_argument('--cutoff_dist', default=5.0, type=float)

    # Training
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)

    return parser.parse_args()


def main():
    t_start = time.time()
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')

    args = cli()
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[{time.strftime('%H:%M:%S')}] Using device: {device}")

    # Prepare data
    print(f"[{time.strftime('%H:%M:%S')}] Preparing data...")
    dm = QM9DataModule(
        target=args.target,
        data_dir=args.data_dir,
        batch_size_train=args.batch_size_train,
        batch_size_inference=args.batch_size_inference,
        num_workers=args.num_workers,
        splits=args.splits,
        seed=args.seed,
        subset_size=args.subset_size,
    )
    dm.prepare_data()
    dm.setup()
    y_mean, y_std, atom_refs = dm.get_target_stats(
        remove_atom_refs=True, divide_by_atoms=True
    )

    # Create model
    painn = PaiNN(
        num_message_passing_layers=args.num_message_passing_layers,
        num_features=args.num_features,
        num_outputs=args.num_outputs,
        num_rbf_features=args.num_rbf_features,
        num_unique_atoms=args.num_unique_atoms,
        cutoff_dist=args.cutoff_dist,
    )
    post_processing = AtomwisePostProcessing(
        args.num_outputs, y_mean, y_std, atom_refs
    )

    painn.to(device)
    post_processing.to(device)

    optimizer = torch.optim.AdamW(
        painn.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Save results
    runs_dir = "runs"

    # Detect which hyperparameter was changed
    experiment_params = {}
    defaults = {
        "target": 7,
        "num_message_passing_layers": 3,
        "lr": 5e-4,
        "num_features": 128,
        "num_rbf_features": 20,
        "cutoff_dist": 5.0,
    }

   
    experiment_params["target"] = args.target
    experiment_params["layers"] = args.num_message_passing_layers
    experiment_params["lr"] = args.lr
    experiment_params["features"] = args.num_features
    experiment_params["rbf"] = args.num_rbf_features
    experiment_params["cutoff"] = args.cutoff_dist


    variable_name = "_".join([f"{k}_{v}" for k, v in experiment_params.items()])


    run_folder = os.path.join(runs_dir, f"{run_timestamp}_{variable_name}")
    os.makedirs(run_folder, exist_ok=True)

    # Training
    train_losses_per_epoch = []

    painn.train()
    pbar = trange(args.num_epochs)
    for epoch in pbar:
        loss_epoch = 0.
        for batch in dm.train_dataloader():
            batch = batch.to(device)

            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            loss_step = F.mse_loss(preds, batch.y, reduction='sum')

            loss = loss_step / len(batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss_step.detach().item()
        loss_epoch /= len(dm.data_train)
        train_losses_per_epoch.append(loss_epoch)
        pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}')

    # Evaluation
    painn.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            batch = batch.to(device)

            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch,
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )

            all_preds.append(preds.cpu())
            all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    per_sample_abs_errors = torch.abs(all_preds.squeeze() - all_targets.squeeze())

    # Final MAE
    mae = per_sample_abs_errors.mean()
    unit_conversion = dm.unit_conversion[args.target]
    final_mae = float(unit_conversion(mae))

    print(f"[{time.strftime('%H:%M:%S')}] Test MAE: {final_mae:.3f}")

    # Save model
    model_save_path = os.path.join(run_folder, "trained_painn_model.pt")
    torch.save(painn.state_dict(), model_save_path)
    print(f"[{time.strftime('%H:%M:%S')}] Model saved to {model_save_path}")

    # Save train loss per epoch
    train_loss_path = os.path.join(run_folder, "train_loss_per_epoch.json")
    with open(train_loss_path, 'w') as f:
        json.dump(train_losses_per_epoch, f, indent=4)
    print(f"[{time.strftime('%H:%M')}] Train losses saved to {train_loss_path}")

    # Calculate model size
    model_size_mb = os.path.getsize(model_save_path) / 1e6

    # Save training summary
    training_info = {
        "Test_MAE": final_mae,
        "Total_time_seconds": round(time.time() - t_start, 2),
        "Best_train_loss": min(train_losses_per_epoch),
        "Best_epoch": train_losses_per_epoch.index(min(train_losses_per_epoch)),
        "Model_size_MB": round(model_size_mb, 2),
        "Data_dir": args.data_dir,
        "Subset_size": args.subset_size,
        "Splits": args.splits,
        "Num_epochs": args.num_epochs,
        "Batch_size_train": args.batch_size_train,
        "Batch_size_inference": args.batch_size_inference,
        "Learning_rate": args.lr,
        "Weight_decay": args.weight_decay,
        "Num_message_passing_layers": args.num_message_passing_layers,
        "Num_features": args.num_features,
        "Num_outputs": args.num_outputs,
        "Num_rbf_features": args.num_rbf_features,
        "Num_unique_atoms": args.num_unique_atoms,
        "Cutoff_distance": args.cutoff_dist,
        "Target": args.target,
        "Target_name": dm.target_types[args.target],
    }

    info_save_path = os.path.join(run_folder, "training_summary.json")
    with open(info_save_path, 'w') as f:
        json.dump(training_info, f, indent=4)
    print(f"[{time.strftime('%H:%M')}] Training summary saved to {info_save_path}")

    # Save per-molecule errors
    per_molecule_errors = {
        "molecule_indices": list(range(len(per_sample_abs_errors))),
        "abs_errors": per_sample_abs_errors.tolist(),
    }
    errors_save_path = os.path.join(run_folder, "per_molecule_errors.json")
    with open(errors_save_path, 'w') as f:
        json.dump(per_molecule_errors, f, indent=4)
    print(f"[{time.strftime('%H:%M')}] Per-molecule errors saved to {errors_save_path}")

    print(f"[{time.strftime('%H:%M')}] Total script time: {time.time() - t_start:.2f} seconds")


if __name__ == '__main__':
    main()
