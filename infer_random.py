import torch
import os
import json
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from config import CONFIG, device
from model import CRNN
from dataset import CharacterEncoder
from utils import ctc_decode


def infer_random_images(num_images=10):
    print(f"Loading encoder from {CONFIG['train_json']}")
    if not os.path.exists(CONFIG['train_json']):
        print(f"Train json not found at {CONFIG['train_json']}")
        return
    with open(CONFIG['train_json'], 'r', encoding='utf-8') as f:
        train_labels = json.load(f)
    encoder = CharacterEncoder(train_labels)
    model = CRNN(encoder.num_classes(), hidden_size=256, pretrained=CONFIG['use_pretrained']).to(device)
    best_model_path = os.path.join('best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"Model file not found at {best_model_path}. Please train the model first.")
        return
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {best_model_path}")
    test_img_dir = 'dataset/dataset/test/images'
    if not os.path.exists(test_img_dir):
        print(f"Test directory {test_img_dir} not found.")
        return
    all_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_images:
        print("No images found in test directory.")
        return
    num_to_select = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_height'], CONFIG['img_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"\nPredicting {len(selected_images)} random images...")
    cols = 2
    rows = math.ceil(len(selected_images) / cols)
    plt.figure(figsize=(15, 3 * rows))
    with torch.no_grad():
        for i, img_name in enumerate(selected_images):
            img_path = os.path.join(test_img_dir, img_name)
            try:
                original_image = Image.open(img_path).convert('RGB')
                image_tensor = transform(original_image).unsqueeze(0).to(device)
                output = model(image_tensor)
                pred_text = ctc_decode(output, encoder)[0]
                print(f"[{i+1}/{num_to_select}] Image: {img_name} -> Prediction: {pred_text}")
                plt.subplot(rows, cols, i + 1)
                plt.imshow(original_image)
                plt.title(f"Pred: {pred_text}", fontsize=12)
                plt.axis('off')
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    output_plot_path = 'random_inference_results.png'
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"\nVisualization saved to {output_plot_path}")

    
if __name__ == "__main__":
    infer_random_images(15)
