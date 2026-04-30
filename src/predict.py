import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image
import toml

from src.datasets import BufferedSlippyMapDirectory
from src.unet import UNet
from src.transforms import ConvertImageMode, ImageToTensor
from src.colors import make_palette

def predict(tiles_dir, mask_dir, tile_size, device, chkpt, num_classes=5):
    # load device - updated to support 5 classes
    net = UNet(num_classes).to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    # preprocess and load
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])

    # tiles file, need to get it again, or do we really need it? why not just predict
    directory = BufferedSlippyMapDirectory(tiles_dir, transform=transform, size=tile_size)
    assert len(directory) > 0, "at least one tile in dataset"

    # loading data
    loader = DataLoader(directory, batch_size=1)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for images, tiles in tqdm(loader, desc="Eval", unit="batch", ascii=True):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))

                prob = directory.unbuffer(prob)
                mask = np.argmax(prob, axis=0)  # This gives class indices: 0, 1, 2, 3, 4
                
                # Keep mask as class indices (0-4) for palette mode
                mask = mask.astype(np.uint8)

                # Create palette for 5 classes (background + 4 classes)
                palette = make_palette("light","green", "yellow","dark","red")  # Make sure this supports 5 classes
                out = Image.fromarray(mask, mode="P")
                out.putpalette(palette)

                os.makedirs(os.path.join(mask_dir, str(z), str(x)), exist_ok=True)
                path = os.path.join(mask_dir, str(z), str(x), str(y) + ".png")
                out.save(path, optimize=True)
    
    print("Prediction Done, saved masks to " + mask_dir)
    print(f"Masks contain class indices: 0 (background), 1-4 (classes)")

if __name__=="__main__":
    config = toml.load('config/predict-config.toml')
    
    city_name = config["city_name"]
    target_type = config["target_type"]
    tiles_dir = os.path.join("results", '02Images', city_name)
    mask_dir = os.path.join("results", "03Masks", target_type, city_name)
    
    tile_size =  config["img_size"]
    num_classes = config.get("num_classes", 5)  # Default to 5 classes

    # load checkpoints
    device = torch.device("cuda")
    if target_type == "Solar":
        checkpoint_path = config["checkpoint_path"]
        checkpoint_name = config["solar_checkpoint"]
        chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)
    
    elif target_type == "Green":
        checkpoint_path = config["checkpoint_path"]
        checkpoint_name = config["green_checkpoint"]
        chkpt = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)

    
    predict(tiles_dir, mask_dir, tile_size, device, chkpt, num_classes)