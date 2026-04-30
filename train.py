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
from src.unet import UNet
from src.utils import plot
from src.train import get_dataset_loaders, train, validate

def loop():
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        #sys.exit("Error: CUDA requested but not available")

    # Calculated weights for 5 classes based on inverse frequency
    # Class 0 (background): 0.044 - very common
    # Class 1: 1.578 - less common
    # Class 2: 1.041 - moderately common
    # Class 3: 1.596 - less common
    # Class 4: 0.742 - more common
    weight = torch.Tensor([0.06660202394031003, 1.649068049132022, 0.8240607221382403, 1.8162985008169406, 0.6439707039724872])
    
    if loss_func not in ("CrossEntropy", "mIoU", "Focal", "Lovasz"):
        sys.exit("Error: Unknown Loss Function value!")

    # loading Model with 5 classes
    net = UNet(num_classes)
    net = DataParallel(net)
    net = net.to(device)

    # define optimizer 
    optimizer = Adam(net.parameters(), lr=lr)

    # resume training
    if model_path:
        print(f"Loading checkpoint from {model_path}")
        chkpt = torch.load(model_path, map_location=device)
        net.load_state_dict(chkpt["state_dict"])
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

    # loading data
    train_loader, val_loader = get_dataset_loaders(target_size, batch_size, dataset_path)
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
        print(f"\nTrain - Loss: {train_hist['loss']:.4f}, mIoU: {train_hist['miou']:.3f}, MCC: {train_hist['mcc']:.3f}")
        print(f"Val   - Loss: {val_hist['loss']:.4f}, mIoU: {val_hist['miou']:.3f}, MCC: {val_hist['mcc']:.3f}")
        
        # Print per-class IoU if available
        if "iou_per_class" in train_hist:
            print(f"Train IoU per class: {[f'{iou:.3f}' for iou in train_hist['iou_per_class']]}")
        if "iou_per_class" in val_hist:
            print(f"Val   IoU per class: {[f'{iou:.3f}' for iou in val_hist['iou_per_class']]}")
        
        # Store history
        for key, value in train_hist.items():
            history["train " + key].append(value)

        for key, value in val_hist.items():
            history["val " + key].append(value)

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

if __name__ == "__main__":
    config = toml.load('config/train-config.toml')

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