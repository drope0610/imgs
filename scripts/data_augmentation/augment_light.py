import os
import random
from PIL import Image, ImageEnhance

input_dir = "/Users/pedro/Desktop/Inria/imgs/datasets/mvtec_anomaly_detection/capsule/train/good"
output_dir = "/Users/pedro/Desktop/Inria/imgs/datasets/mvtec_anomaly_detection/capsule/train/modified"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        
        # Create a slight lighting modification
        # Brightness factor: 1.0 is original, < 1.0 is darker, > 1.0 is brighter
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        img_modified = enhancer.enhance(factor)
        
        out_path = os.path.join(output_dir, filename)
        img_modified.save(out_path)
        print(f"Saved {filename} with brightness factor {factor:.2f}")

print("Done generating modified images.")
