import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CharacterEncoder:
    def __init__(self, labels_dict):
        all_chars = set()
        for text in labels_dict.values():
            all_chars.update(text)
        chars = sorted(list(all_chars))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.char_to_idx['BLANK'] = 0
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        print(f"Total unique characters: {len(chars)}")
    def encode(self, text):
        return [self.char_to_idx.get(char, 0) for char in text]
    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx != 0 and idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)
    def num_classes(self):
        return len(self.char_to_idx)
    

class OCRDataset(Dataset):
    def __init__(self, json_path, img_dir, encoder, transform=None, img_height=32, img_width=256):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.encoder = encoder
        self.image_paths = list(self.data.keys())
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = self.img_dir / img_name
        text = self.data[img_name]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (self.img_width, self.img_height), color='white')
        if self.transform:
            image = self.transform(image)
        encoded_text = self.encoder.encode(text)
        return image, torch.LongTensor(encoded_text), text
    

def get_transforms(img_height, img_width):
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomAffine(degrees = 10, translate=(0.1, 0.1)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomGrayscale(p = 0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def collate_fn(x):
    return (
        torch.stack([item[0] for item in x]),
        [item[1] for item in x],
        [item[2] for item in x]
    )
