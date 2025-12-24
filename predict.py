import torch
import os
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from config import CONFIG, device
from model import CRNN
from dataset import CharacterEncoder
from utils import ctc_decode


def predict_and_save(model, encoder, img_dir, output_json, device, img_height=32, img_width=1024):
    model.eval()
    predictions = {}
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return
    img_names = sorted(os.listdir(img_dir))
    print(f"Predicting {len(img_names)} images from: {img_dir}")
    with torch.no_grad():
        for img_name in tqdm(img_names):
            img_path = os.path.join(img_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            image = transform(image).unsqueeze(0).to(device)
            output = model(image)
            pred_text = ctc_decode(output, encoder)[0]
            predictions[img_name] = pred_text.strip()
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to: {output_json}")


def main():
    print(f"Loading encoder from {CONFIG['train_json']}")
    with open(CONFIG['train_json'], 'r', encoding='utf-8') as f:
        train_labels = json.load(f)
    encoder = CharacterEncoder(train_labels)
    model = CRNN(encoder.num_classes(), hidden_size=256, pretrained=CONFIG['use_pretrained']).to(device)
    best_model_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model from: {best_model_path}")
        test_dir = 'dataset/dataset/test/images'
        if os.path.exists(test_dir):
             predict_and_save(
                model, encoder,
                img_dir=test_dir,
                output_json='./test_predictions.json',
                device=device,
                img_height=CONFIG['img_height'],
                img_width=CONFIG['img_width']
            )
        else:
            print(f"Test directory {test_dir} not found.")
    else:
        print("Best model not found. Skipping prediction.")

        
if __name__ == '__main__':
    main()
