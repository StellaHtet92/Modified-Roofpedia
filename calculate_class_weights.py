import numpy as np
from PIL import Image
from pathlib import Path

def calculate_class_weights(mask_dir, num_classes=5):
    """Calculate inverse frequency weights for classes."""
    class_counts = np.zeros(num_classes)
    
    mask_files = list(Path(mask_dir).rglob("*.png"))
    
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file).convert('P'))
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    # Calculate inverse frequency weights
    total_pixels = class_counts.sum()
    weights = total_pixels / (num_classes * class_counts)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    print("Class counts:", class_counts)
    print("Class weights:", weights)
    print("\nAdd this to your train.py:")
    print(f"weight = torch.Tensor({weights.tolist()})")
    
    return weights

# Run this on your training masks
calculate_class_weights("dataset/labels")