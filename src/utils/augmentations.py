import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

class RandomInpaintReflection(torch.nn.Module):
    """
    Applique une simulation de suppression de reflet aléatoire via OpenCV Inpainting.
    Fonctionne sur des PIL Images ou des Tensors PyTorch.
    """
    def __init__(self, p=0.5, threshold=225, attenuation_only=False):
        super().__init__()
        self.p = p
        self.threshold = threshold
        self.attenuation_only = attenuation_only

    def forward(self, img):
        if torch.rand(1).item() > self.p:
            return img

        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # Convert Tensor (C,H,W) to numpy (H,W,C) [0, 255] BGR
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        if self.attenuation_only:
            # Attenuate bright spots
            img_np = img_np.astype(np.float32)
            img_np[mask == 255] = img_np[mask == 255] * 0.6
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            # Full removal via Telea Inpainting
            kernel = np.ones((3,3), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            img_np = cv2.inpaint(img_np, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        if is_tensor:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        else:
            return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

class SyntheticNoiseInjection(torch.nn.Module):
    """
    Ajoute de légères imperfections artificielles (bruit gaussien, petites tâches).
    Le but est d'apprendre au modèle à ne pas sur-réagir aux variations mineures (baisse du Taux de Faux Positifs).
    """
    def __init__(self, p=0.4, max_noise_std=0.03):
        super().__init__()
        self.p = p
        self.max_noise_std = max_noise_std

    def forward(self, img):
        if torch.rand(1).item() > self.p:
            return img
            
        is_tensor = isinstance(img, torch.Tensor)
        if not is_tensor:
            img = T.functional.to_tensor(img)
            
        # 1. Bruit Gaussien léger
        noise_std = torch.rand(1).item() * self.max_noise_std
        noise = torch.randn_like(img) * noise_std
        img = img + noise
        
        # 2. Petites variations d'intensité aléatoires (simulation de poussières sombres ou claires minuscules)
        if torch.rand(1).item() > 0.5:
            num_dust = torch.randint(1, 5, (1,)).item()
            c, h, w = img.shape
            for _ in range(num_dust):
                y, x = torch.randint(0, h, (1,)).item(), torch.randint(0, w, (1,)).item()
                intensity_shift = (torch.rand(1).item() - 0.5) * 0.15 # +/- 15%
                # Appliquer à un petit patch de 3x3
                img[:, max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] += intensity_shift
        
        img = torch.clamp(img, 0.0, 1.0)
        
        if not is_tensor:
            img = T.functional.to_pil_image(img)
            
        return img

def get_train_augmentations(img_size=256):
    """
    Retourne la pipeline de Data Augmentation complète.
    Inclut:
    - Redimensionnement
    - Random Brightness (0.7 à 1.3)
    - Random Reflection Attenuation (p=0.3)
    - Random Reflection Inpainting (p=0.3)
    - Synthetic Noise Injection (p=0.4)
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=(0.7, 1.3)),
        RandomInpaintReflection(p=0.3, attenuation_only=True),
        RandomInpaintReflection(p=0.3, attenuation_only=False),
        SyntheticNoiseInjection(p=0.4, max_noise_std=0.02),
        T.ToTensor()
    ])
