import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import os
import editdistance
from config import CONFIG, device
from model import CRNN
from dataset import OCRDataset, CharacterEncoder, get_transforms, collate_fn
from utils import ctc_decode


def train_epoch(model, dataloader, criterion, optimizer, device, encoder, epoch):
    if not hasattr(train_epoch, "scaler"):
        train_epoch.scaler = GradScaler()
    scaler = train_epoch.scaler
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', ncols=100)
    for images, targets, target_texts in pbar:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full(
                (outputs.size(1),),
                outputs.size(0),
                dtype=torch.long,
                device=device
            )
            target_lengths = torch.LongTensor([len(t) for t in targets]).to(device)
            targets_flat = torch.cat(targets).to(device)
            loss = criterion(
                outputs.log_softmax(2),
                targets_flat,
                input_lengths,
                target_lengths
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, encoder, dataset_name="Valid"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_sent = 0
    total_edit_dist = 0.0
    pbar = tqdm(dataloader, desc=f'{dataset_name:>7}', ncols=100, leave=False)
    with torch.no_grad():
        for images, targets, target_texts in pbar:
            images = images.to(device)
            outputs = model(images)
            outputs_for_loss = outputs.permute(1, 0, 2)
            input_lengths = torch.full((outputs_for_loss.size(1),), outputs_for_loss.size(0), dtype=torch.long)
            target_lengths = torch.LongTensor([len(t) for t in targets])
            targets_flat = torch.cat(targets)
            loss = criterion(outputs_for_loss.log_softmax(2), targets_flat, input_lengths, target_lengths)
            total_loss += loss.item()
            predictions = ctc_decode(outputs, encoder)
            for pred, true in zip(predictions, target_texts):
                pred, true = pred.strip(), true.strip()
                total_samples += 1
                if pred == true:
                    correct_sent += 1
                total_edit_dist += editdistance.eval(pred, true)
            current_acc = 100 * correct_sent / total_samples if total_samples > 0 else 0
            pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
    avg_loss = total_loss / len(dataloader)
    sent_acc = 100 * correct_sent / total_samples if total_samples > 0 else 0
    avg_lev_dist = total_edit_dist / total_samples if total_samples > 0 else 0
    return avg_loss, sent_acc, avg_lev_dist


def main():
    print(f"Loading data from {CONFIG['train_json']}")
    with open(CONFIG['train_json'], 'r', encoding='utf-8') as f:
        train_labels = json.load(f)
    encoder = CharacterEncoder(train_labels)
    train_transform, val_transform = get_transforms(CONFIG['img_height'], CONFIG['img_width'])
    train_dataset = OCRDataset(CONFIG['train_json'], CONFIG['train_img_dir'], encoder, train_transform)
    valid_dataset = OCRDataset(CONFIG['valid_json'], CONFIG['valid_img_dir'], encoder, val_transform)
    print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)}\n")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'], collate_fn=collate_fn, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                              num_workers=CONFIG['num_workers'], collate_fn=collate_fn, pin_memory=True)
    model = CRNN(encoder.num_classes(), hidden_size=256, pretrained=CONFIG['use_pretrained']).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_valid_acc = 0
    print("Starting training...\n")
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, encoder, epoch)
        valid_loss, valid_acc, valid_lev = validate(model, valid_loader, criterion, device, encoder)
        scheduler.step(valid_loss)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Valid Acc: {valid_acc:.2f}% | Valid Loss: {valid_loss:.4f}")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            save_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model (Acc: {best_valid_acc:.2f}%)")
    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_valid_acc:.2f}%")

    
if __name__ == '__main__':
    main()
