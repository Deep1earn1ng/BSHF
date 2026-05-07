import random
import math
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
from torchvision import transforms as T

class RandomGammaCorrection(object):
    def __init__(self, p=0.5, gamma_limit=(0.5, 1.5)):
        self.p = p
        self.gamma_limit = gamma_limit

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(self.gamma_limit[0], self.gamma_limit[1])
            img = F.adjust_gamma(img, gamma=gamma, gain=1)
        return img

class RandomWhiteBalance(object):
    def __init__(self, p=0.5, gain_variance=0.2):
        self.p = p
        self.gain_variance = gain_variance

    def __call__(self, img):
        if random.random() < self.p and isinstance(img, Image.Image):
            r, g, b = img.split()
            r_gain = random.uniform(1.0 - self.gain_variance, 1.0 + self.gain_variance)
            b_gain = random.uniform(1.0 - self.gain_variance, 1.0 + self.gain_variance)
            r = r.point(lambda i: i * r_gain)
            b = b.point(lambda i: i * b_gain)
            img = Image.merge('RGB', (r, g, b))
        return img

class LSEA(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.random() < self.p:
            img_w, img_h = img.size
            area = img_w * img_h
            for _ in range(100):
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    x1 = random.randint(0, img_w - w)
                    y1 = random.randint(0, img_h - h)
                    patch = img.crop((x1, y1, x1 + w, y1 + h))
                    patch_gray = ImageOps.grayscale(patch).convert('RGB')
                    img.paste(patch_gray, (x1, y1, x1 + w, y1 + h))
                    return img
        return img

class PatchRandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406], patch_size=16):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.patch_size = patch_size

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            h = (h // self.patch_size) * self.patch_size
            w = (w // self.patch_size) * self.patch_size
            
            if h == 0 or w == 0:
                continue

            if w < img.size()[2] and h < img.size()[1]:
                max_x = (img.size()[1] - h) // self.patch_size
                max_y = (img.size()[2] - w) // self.patch_size
                
                if max_x <= 0 or max_y <= 0:
                    continue
                    
                x1 = random.randint(0, max_x) * self.patch_size
                y1 = random.randint(0, max_y) * self.patch_size
                
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

def build_transforms(cfg, is_train=True):
    res = []
    if is_train:
        res.append(T.Resize((cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])))
        res.append(T.RandomHorizontalFlip(p=0.5))
        res.append(T.Pad(cfg.INPUT.PADDING))
        res.append(T.RandomCrop((cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])))
    else:
        res.append(T.Resize((cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])))

    if is_train:
        res.append(RandomGammaCorrection(p=0.5))
        res.append(RandomWhiteBalance(p=0.5))
        res.append(LSEA(p=0.5))

    res.append(T.ToTensor())
    res.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))

    if is_train:
        res.append(PatchRandomErasing(probability=0.5, mean=cfg.INPUT.PIXEL_MEAN))

    return T.Compose(res)