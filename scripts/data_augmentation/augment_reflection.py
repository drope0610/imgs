import os
import cv2
import numpy as np

input_dir = "/Users/pedro/Desktop/Inria/imgs/datasets/mvtec_anomaly_detection/capsule/train/good"
out_attenuated_dir = "/Users/pedro/Desktop/Inria/imgs/datasets/mvtec_anomaly_detection/capsule/train/reflection_attenuated"
out_removed_dir = "/Users/pedro/Desktop/Inria/imgs/datasets/mvtec_anomaly_detection/capsule/train/reflection_removed"

os.makedirs(out_attenuated_dir, exist_ok=True)
os.makedirs(out_removed_dir, exist_ok=True)

# Threshold to detect reflections (very bright pixels)
# You may need to adjust this value (0-255) depending on the exact brightness of the reflection
REFLECTION_THRESHOLD = 225

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # Convert to grayscale to find bright spots easily
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create a mask of the reflection
        _, mask = cv2.threshold(gray, REFLECTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 1. Attenuate the reflection
        # We reduce the intensity of pixels where mask is 255
        img_attenuated = img.copy().astype(np.float32)
        
        # Reduce brightness in the mask area by multiplying by 0.6
        attenuation_factor = 0.6
        img_attenuated[mask == 255] = img_attenuated[mask == 255] * attenuation_factor
        
        # Ensure values stay in valid range
        img_attenuated = np.clip(img_attenuated, 0, 255).astype(np.uint8)
        
        # 2. Remove the reflection using Inpainting
        # We dilate the mask slightly to make sure the edges of the reflection are covered
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply Telea inpainting
        img_removed = cv2.inpaint(img, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Save both results
        cv2.imwrite(os.path.join(out_attenuated_dir, filename), img_attenuated)
        cv2.imwrite(os.path.join(out_removed_dir, filename), img_removed)
        print(f"Processed {filename}")

print("Done generating images with modified reflections.")
