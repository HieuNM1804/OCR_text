import torch
import os


CONFIG = {
    'train_json': 'dataset/dataset/train.json',
    'train_img_dir': 'dataset/dataset/train/images',
    'valid_json': 'dataset/dataset/valid.json',
    'valid_img_dir': 'dataset/dataset/valid/images',
    'img_height': 128,
    'img_width': 1024,
    'batch_size': 8,
    'num_epochs': 60,
    'learning_rate': 0.0002,
    'num_workers': 8,
    'save_dir': './output/',
    'use_pretrained': True,
}


os.makedirs(CONFIG['save_dir'], exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')
