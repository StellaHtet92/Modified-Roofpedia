import os
import sys
import collections
import toml
from tqdm import tqdm
import webp
import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop, Normalize

from src.losses import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from src.unet_fusion import FusionUNet as UNet
from src.utils import plot
from src.train_multimodal import get_dataset_loaders, train, validate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def loop():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        #sys.exit("Error: CUDA requested but not available")

    # Calculated weights for 5 classes based on inverse frequency
    # Class 0 (background): 0.044 - very common
    # Class 1: 1.578 - less common
    # Class 2: 1.041 - moderately common
    # Class 3: 1.596 - less common
    # Class 4: 0.742 - more common
    weight = torch.Tensor([0.04973079, 0.51461095, 0.86000951, 2.44157207, 1.13407668])
    
    if loss_func not in ("CrossEntropy", "mIoU", "Focal", "Lovasz"):
        sys.exit("Error: Unknown Loss Function value!")

    # loading Model with 5 classes
    net = UNet(num_classes=num_classes, struct_channels=4)
    net = DataParallel(net)
    net = net.to(device)

    # define optimizer 
    optimizer = Adam(net.parameters(), lr=lr)

    # resume training
    if model_path:
        print(f"Loading checkpoint from {model_path}")
        chkpt = torch.load(model_path, map_location=device, weights_only=False)
        missing, unexpected = net.load_state_dict(chkpt["state_dict"], strict=False)
        if missing or unexpected:
            # e.g. resuming from a checkpoint saved before ConvRelu had BatchNorm:
            # the new bn.* keys are "missing" and start from scratch. The optimizer's
            # saved state was built for the OLD parameter list/order, so it no longer
            # lines up with net.parameters() -- loading it here would silently apply
            # stale per-parameter Adam state to the wrong parameters. Skip it instead.
            print(f"WARNING: model architecture differs from this checkpoint "
                  f"(missing={len(missing)}, unexpected={len(unexpected)} keys). "
                  f"New/changed layers start from scratch. NOT resuming optimizer "
                  f"state since it no longer matches the model's parameter list.")
        else:
            optimizer.load_state_dict(chkpt["optimizer"])
        print(f"Resumed from epoch {chkpt.get('epoch', 'unknown')}")

    # select loss function
    if loss_func == "CrossEntropy":
        criterion = CrossEntropyLoss2d(weight=weight).to(device)
    elif loss_func == "mIoU":
        criterion = mIoULoss2d(weight=weight).to(device)
    elif loss_func == "Focal":
        criterion = FocalLoss2d(weight=weight).to(device)
    elif loss_func == "Lovasz":
        criterion = LovaszLoss2d().to(device)

    print(f"Using loss function: {loss_func}")
    print(f"Class weights: {weight.tolist()}")
    print(f"Fusion: SpatialAttentionFusion (both modalities always combined; "
          f"no attention regularizer needed -- see src/unet_fusion.py docstring)")

    # loading data
    if struct_mean is not None and struct_std is not None:
        print(f"Using fixed struct stats from config: mean={struct_mean} std={struct_std}")
    else:
        print("No struct_mean/struct_std in config -- recomputing from a random "
              "200-file sample of the training struct folder (see [struct stats] "
              "line below; not guaranteed stable across reruns/machines).")
    train_loader, val_loader = get_dataset_loaders(
        target_size, batch_size, dataset_path,
        struct_mean=struct_mean, struct_std=struct_std,
    )
    history = collections.defaultdict(list)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # training loop
    for epoch in range(0, num_epochs):

        print("\n" + "="*60)
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print("="*60)
        
        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        val_hist = validate(val_loader, num_classes, device, net, criterion)

        # Print metrics
        # (mean_IoU was removed here: it read train_hist['iou_per_class'], a key
        # train()/validate() never actually return -- Metrics (src/metrics.py) only
        # tracks binary tn/fn/fp/tp, not per-class IoU -- so it always printed 0.000
        # regardless of real performance. Trust mIoU/MCC above instead.)
        print(f"\nTrain - Loss: {train_hist['loss']:.4f}, mIoU: {train_hist['miou']:.3f}, MCC: {train_hist['mcc']:.3f}")
        print(f"Val   - Loss: {val_hist['loss']:.4f}, mIoU: {val_hist['miou']:.3f}, MCC: {val_hist['mcc']:.3f}")
        # Diagnostic only (not regularized toward any value): mean spatial gate activation.
        print(f"Attn  - train attn_mean={train_hist['attn_mean']:.4f}  |  val attn_mean={val_hist['attn_mean']:.4f}")
        # Print per-class IoU if available
        if "iou_per_class" in train_hist:
            print(f"Train IoU per class: {[f'{iou:.3f}' for iou in train_hist['iou_per_class']]}")
        if "iou_per_class" in val_hist:
            print(f"Val   IoU per class: {[f'{iou:.3f}' for iou in val_hist['iou_per_class']]}")
        
        # Store history
        for key, value in train_hist.items():
            history["train " + key].append(value)
        if "iou_per_class" in train_hist:
            history["train mean_iou"].append(float(np.mean(train_hist["iou_per_class"])))

        for key, value in val_hist.items():
            history["val " + key].append(value)
        if "iou_per_class" in val_hist:
            history["val mean_iou"].append(float(np.mean(val_hist["iou_per_class"])))

        # Save visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            visual = "history-{:05d}-of-{:05d}.png".format(epoch + 1, num_epochs)
            plot(os.path.join(checkpoint_path, visual), history)
            print(f"Saved training plot: {visual}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint = target_type + "-checkpoint-{:03d}-of-{:03d}.pth".format(epoch + 1, num_epochs)
            states = {
                "epoch": epoch + 1, 
                "state_dict": net.state_dict(), 
                "optimizer": optimizer.state_dict(),
                "history": dict(history)
            }
            torch.save(states, os.path.join(checkpoint_path, checkpoint))
            print(f"Saved checkpoint: {checkpoint}")

    # Save final model
    final_checkpoint = target_type + "-final.pth"
    states = {
        "epoch": num_epochs, 
        "state_dict": net.state_dict(),                
        "optimizer": optimizer.state_dict(),
        "history": dict(history)
    }
    torch.save(states, os.path.join(checkpoint_path, final_checkpoint))
    print(f"\nTraining complete! Final model saved: {final_checkpoint}")

    # ── Save history as CSV ───────────────────────────────────
    history_scalar = {
        k: v for k, v in history.items()
        if not isinstance(v[0], (list, np.ndarray))
    }
    history_df = pd.DataFrame(history_scalar)
    csv_path   = os.path.join(checkpoint_path, target_type + "-history.csv")
    history_df.to_csv(csv_path, index_label="epoch")
    print(f"History saved: {csv_path}")

    # ── Save one final plot per metric ────────────────────────
    metric_names = set(
        k.replace("train ", "").replace("val ", "") for k in history.keys()
        if "iou_per_class" not in k
    )
    metrics_dir = os.path.join(checkpoint_path, "metric_plots")
    os.makedirs(metrics_dir, exist_ok=True)

    for metric in metric_names:
        train_vals = history.get("train " + metric, [])
        val_vals   = history.get("val "   + metric, [])

        fig, ax = plt.subplots(figsize=(10, 5))
        epochs  = range(1, len(train_vals) + 1)   # ← use actual length, not num_epochs

        if train_vals:
            ax.plot(epochs, train_vals, label=f"Train {metric}", linewidth=2)
        if val_vals:
            ax.plot(epochs, val_vals, label=f"Val {metric}", linewidth=2, linestyle="--")

        ax.set_title(metric.upper(), fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"{target_type}-{metric.replace(' ', '_')}-final.png"
        fig.savefig(os.path.join(metrics_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {fname}")


if __name__ == "__main__":
    config = toml.load('D:/DeepLearning/Roofpedia_IEEE_Paper/Modified_Roofpedia_fusion_with_mapbox/config/train-config.toml')

    num_classes = 5  # 5 classes: background + 4 object classes
    lr = config['lr']
    loss_func = config['loss_func']
    num_epochs = config['num_epochs']
    target_size = config['target_size']
    batch_size = config['batch_size']

    dataset_path = config['dataset_path']
    checkpoint_path = config['checkpoint_path']
    target_type = config['target_type']

    model_path = None
    if config.get('model_path', '') != '':
        model_path = config['model_path']

    # Fixed struct [nDSM, slope, aspect_sin, aspect_cos] normalisation stats. If absent,
    # get_dataset_loaders() falls back to estimating them from a random 200-file sample
    # of the training struct folder every run (see compute_struct_stats in
    # src/train_multimodal.py) -- cheap, but not guaranteed identical across reruns or
    # machines (os.listdir order isn't guaranteed stable), and the checkpoint itself
    # does not store whatever values were used. Setting them here makes normalisation
    # reproducible and gives you one canonical set of numbers to reuse in
    # inspect_fusion.py / prediction-time normalisation for checkpoints trained with
    # this config.
    struct_mean = config.get('struct_mean', None)
    struct_std = config.get('struct_std', None)

    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Target type: {target_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Learning rate: {lr}")
    print(f"Loss function: {loss_func}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Target size: {target_size}")
    print(f"Checkpoint path: {checkpoint_path}")
    if model_path:
        print(f"Resume from: {model_path}")
    print("="*60 + "\n")

    # make dir for checkpoint
    os.makedirs(checkpoint_path, exist_ok=True)
    
    loop()