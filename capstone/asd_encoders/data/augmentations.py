
import numpy as np
import torch
import cv2
import random

class EEGAugment:
    def __init__(self, sfreq=250, p=0.5):
        self.sfreq = sfreq
        self.p = p

    def gaussian_noise(self, x, std_frac=0.02):
        std = std_frac * (np.std(x) + 1e-8)
        return x + np.random.randn(*x.shape).astype(np.float32) * std

    def channel_dropout(self, x, max_channels=2):
        x = x.copy()
        n_ch = x.shape[0]
        n_drop = random.randint(0, min(max_channels, n_ch - 1))
        if n_drop > 0:
            idx = np.random.choice(n_ch, n_drop, replace=False)
            x[idx, :] = 0.0
        return x

    def time_mask(self, x, max_frac=0.1):
        x = x.copy()
        t = x.shape[1]
        mask_len = int(t * random.uniform(0, max_frac))
        if mask_len > 0:
            start = random.randint(0, t - mask_len)
            x[:, start:start + mask_len] = 0.0
        return x

    def amplitude_scale(self, x, low=0.9, high=1.1):
        scale = random.uniform(low, high)
        return x * scale

    def time_shift(self, x, max_shift_frac=0.05):
        t = x.shape[1]
        shift = int(t * random.uniform(-max_shift_frac, max_shift_frac))
        return np.roll(x, shift, axis=1)

    def __call__(self, x):
        # x: [channels, time] float32
        if random.random() < self.p:
            x = self.gaussian_noise(x)
        if random.random() < self.p:
            x = self.channel_dropout(x)
        if random.random() < self.p:
            x = self.time_mask(x)
        if random.random() < self.p:
            x = self.amplitude_scale(x)
        if random.random() < self.p:
            x = self.time_shift(x)
        return x.astype(np.float32)


class GazeAugment:
    def __init__(self, img_size=224, p=0.5, train=True):
        self.img_size = img_size
        self.p = p
        self.train = train

    def random_crop_resize(self, img, scale=(0.8, 1.0)):
        h, w = img.shape[:2]
        s = random.uniform(*scale)
        ch, cw = int(h * s), int(w * s)
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        crop = img[top:top + ch, left:left + cw]
        return cv2.resize(crop, (self.img_size, self.img_size))

    def horizontal_flip(self, img):
        return cv2.flip(img, 1)

    def brightness_contrast(self, img, b_range=(-15, 15), c_range=(0.85, 1.15)):
        img = img.astype(np.float32)
        b = random.uniform(*b_range)
        c = random.uniform(*c_range)
        img = img * c + b
        return np.clip(img, 0, 255)

    def gaussian_blur(self, img, k=3):
        return cv2.GaussianBlur(img, (k, k), 0)

    def cutout(self, img, max_frac=0.2):
        h, w = img.shape[:2]
        ch, cw = int(h * random.uniform(0, max_frac)), int(w * random.uniform(0, max_frac))
        if ch > 0 and cw > 0:
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            img = img.copy()
            img[top:top + ch, left:left + cw] = 0
        return img

    def __call__(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32)
        if self.train:
            if random.random() < self.p:
                img = self.random_crop_resize(img)
            if random.random() < self.p:
                img = self.horizontal_flip(img) 
            if random.random() < self.p:
                img = self.brightness_contrast(img)
            if random.random() < 0.3:
                img = self.gaussian_blur(img)
            if random.random() < 0.3:
                img = self.cutout(img)
        img = img / 255.0
        img = (img - 0.5) / 0.5 
        return img.astype(np.float32)


class GaitAugment:
    def __init__(self, p=0.5):
        self.p = p

    def jitter(self, x, std=0.01):
        return x + np.random.randn(*x.shape).astype(np.float32) * std

    def rotate(self, x, max_deg=15):
      
        theta = np.deg2rad(random.uniform(-max_deg, max_deg))
        rot = np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
        c = x.shape[-1]
        if c >= 3:
            x = x.copy()
            x[..., :3] = x[..., :3] @ rot.T
        return x

    def scale(self, x, low=0.9, high=1.1):
        s = random.uniform(low, high)
        return x * s

    def time_warp(self, x, low=0.9, high=1.1):

        t = x.shape[0]
        new_t = max(2, int(t * random.uniform(low, high)))
        idx = np.linspace(0, t - 1, new_t)
        idx_floor = np.floor(idx).astype(int)
        idx_floor = np.clip(idx_floor, 0, t - 1)
        x = x[idx_floor]

        if x.shape[0] != t:
            orig_idx = np.linspace(0, x.shape[0] - 1, t)
            x = np.stack([
                np.interp(orig_idx, np.arange(x.shape[0]), x[:, v, c])
                for v in range(x.shape[1]) for c in range(x.shape[2])
            ], axis=-1).reshape(t, x.shape[1], x.shape[2])
        return x.astype(np.float32)

    def temporal_mask(self, x, max_frac=0.1):
        t = x.shape[0]
        mask_len = int(t * random.uniform(0, max_frac))
        if mask_len > 0:
            start = random.randint(0, t - mask_len)
            x = x.copy()
            x[start:start + mask_len] = 0.0
        return x

    def __call__(self, x):

        if random.random() < self.p:
            x = self.jitter(x)
        if random.random() < self.p:
            x = self.rotate(x)
        if random.random() < self.p:
            x = self.scale(x)
        if random.random() < self.p:
            x = self.time_warp(x)
        if random.random() < self.p:
            x = self.temporal_mask(x)
        return x.astype(np.float32)
