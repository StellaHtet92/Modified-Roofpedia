import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import DataParallel
from torchvision.transforms import Compose, Normalize
from PIL import Image

from src.unet import UNet
from src.train import get_dataset_loaders
from src.transforms import ConvertImageMode, ImageToTensor
from src.colors import Mapbox

def visualize_predictions(model_path, dataset_path, target_size=512, batch_size=4, num_samples=10, save_dir='visualizations'):
    """
    Visualize predictions from a trained model.
    
    Args:
        model_path: Path to the saved checkpoint (e.g., 'checkpoint/Green-final.pth')
        dataset_path: Path to validation dataset
        target_size: Image size for model input
        batch_size: Batch size for data loading
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualization images
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    num_classes = 5
    net = UNet(num_classes)
    net = DataParallel(net)
    
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net = net.to(device)
    net.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load validation data
    print("Loading validation data...")
    train_loader, val_loader = get_dataset_loaders(target_size, batch_size, dataset_path)
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define class colors (matching your 5 classes)
    class_colors = [
        [0, 0, 0],           # Class 0: Background (black)
        Mapbox.red.value,    # Class 1: Red
        Mapbox.green.value,  # Class 2: Green
        Mapbox.blue.value,   # Class 3: Blue
        Mapbox.yellow.value, # Class 4: Yellow
    ]
    
    class_names = ['Background', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    # Denormalization for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    print(f"\nGenerating {num_samples} visualizations...")
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if sample_count >= num_samples:
                break
            
            images = images.to(device)
            outputs = net(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            
            # Process each image in batch
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                # Get image, mask, and prediction
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = std * img + mean  # Denormalize
                img = np.clip(img, 0, 1)
                
                mask = masks[i].cpu().numpy()
                pred = predictions[i]
                
                # Create colored masks
                mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
                pred_colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
                
                for class_idx in range(num_classes):
                    mask_colored[mask == class_idx] = class_colors[class_idx]
                    pred_colored[pred == class_idx] = class_colors[class_idx]
                
                # Create figure with 4 subplots
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                
                # Original Image
                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
                axes[0, 0].axis('off')
                
                # Ground Truth
                axes[0, 1].imshow(mask_colored)
                axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')
                
                # Prediction
                axes[1, 0].imshow(pred_colored)
                axes[1, 0].set_title('Prediction', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
                
                # Overlay: Prediction on Image
                axes[1, 1].imshow(img)
                axes[1, 1].imshow(pred_colored, alpha=0.5)
                axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
                axes[1, 1].axis('off')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                                   for color, name in zip(class_colors, class_names)]
                fig.legend(handles=legend_elements, loc='lower center', 
                          ncol=5, fontsize=10, frameon=True)
                
                plt.tight_layout(rect=[0, 0.03, 1, 1])
                
                # Calculate per-class metrics for this sample
                correct_pixels = (pred == mask)
                overall_acc = correct_pixels.sum() / mask.size
                
                # Add metrics text
                metrics_text = f'Overall Accuracy: {overall_acc:.3f}\n'
                for class_idx in range(num_classes):
                    mask_class = (mask == class_idx)
                    pred_class = (pred == class_idx)
                    if mask_class.sum() > 0:
                        intersection = (mask_class & pred_class).sum()
                        union = (mask_class | pred_class).sum()
                        iou = intersection / union if union > 0 else 0
                        metrics_text += f'{class_names[class_idx]} IoU: {iou:.3f}  '
                
                fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Save figure
                save_path = os.path.join(save_dir, f'sample_{sample_count+1:03d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                print(f"Saved: {save_path}")
    
    print(f"\n✓ Successfully generated {sample_count} visualizations in '{save_dir}/' directory")
    
    # Create a summary grid of all samples
    create_summary_grid(save_dir, sample_count)


def create_summary_grid(save_dir, num_samples):
    """Create a summary grid showing all visualizations."""
    print("\nCreating summary grid...")
    
    # Calculate grid dimensions
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        
        img_path = os.path.join(save_dir, f'sample_{idx+1:03d}.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Sample {idx+1}', fontsize=10)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'summary_grid.png')
    plt.savefig(summary_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary grid saved: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--model', type=str, default='checkpoint/Green-final.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='dataset\validation',
                       help='Path to validation dataset')
    parser.add_argument('--size', type=int, default=512,
                       help='Target image size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for loading data')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    visualize_predictions(
        model_path=args.model,
        dataset_path=args.dataset,
        target_size=args.size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
