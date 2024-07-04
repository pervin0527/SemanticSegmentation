import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import random
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig

from data.bkai import BKAIDataset
from loss import FocalLoss, DiceLoss
from metrics import mean_dice_coefficient
from utils.utils import Args, inference_callback

def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True

def valid(model, dataloader, criterion1, criterion2, device, num_classes, ignore_idx=0):
    model.eval()
    accs, losses, f1_scores, dice_scores = [], [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Eval", leave=False)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            
            loss1 = criterion1(upsampled_logits, labels) # focal loss
            loss2 = criterion2(upsampled_logits, labels) # dice loss
            loss = loss1 + loss2
            losses.append(loss.item()) 

            mask = (labels != ignore_idx)
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels, true_labels)
            accs.append(accuracy)

            f1 = f1_score(true_labels, pred_labels, average='macro')
            f1_scores.append(f1)

            dice_score = mean_dice_coefficient(predicted, labels, num_classes)
            dice_scores.append(dice_score)
            
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_dice = sum(dice_scores) / len(dice_scores)
    
    return avg_loss, avg_acc, avg_f1, avg_dice

def train(model, dataloader, optimizer, criterion1, criterion2, device, num_classes, ignore_idx=0):
    model.train()
    accs, losses, f1_scores, dice_scores = [], [], [], []
    for idx, batch in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)

        loss1 = criterion1(upsampled_logits, labels) # focal loss
        loss2 = criterion2(upsampled_logits, labels) # dice loss
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        mask = (labels != ignore_idx)
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels, true_labels)
        accs.append(accuracy)

        f1 = f1_score(true_labels, pred_labels, average='macro')
        f1_scores.append(f1)

        dice_score = mean_dice_coefficient(predicted, labels, num_classes)
        dice_scores.append(dice_score)

    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_dice = sum(dice_scores) / len(dice_scores)

    return avg_loss, avg_acc, avg_f1, avg_dice

def main(args):
    writer = SummaryWriter(log_dir=f"{args.save_dir}/logs")
    model_config = SegformerConfig.from_pretrained(args.pretrained_model_name,
                                                   id2label=args.id2label, 
                                                   label2id=args.label2id,
                                                   num_labels=len(args.classes),
                                                   image_size=args.img_size,
                                                   num_encoder_blocks=args.num_encoder_blocks,
                                                   drop_path_rate=args.drop_path_rate,
                                                   hidden_dropout_prob=args.hidden_dropout_prob,
                                                   classifier_dropout_prob=args.classifier_dropout_prob,
                                                   attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                                                   semantic_loss_ignore_index=args.semantic_loss_ignore_index)
    model_config.save_pretrained(f'{args.save_dir}')
    feature_extractor = SegformerImageProcessor.from_pretrained(args.pretrained_model_name, do_reduce_labels=args.do_reduce_labels)
    model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_model_name,
                                                             config=model_config,
                                                             ignore_mismatched_sizes=True)
    model.to(args.device)

    train_dataset = BKAIDataset(args, feature_extractor, image_set="train1")
    valid_dataset = BKAIDataset(args, feature_extractor, image_set="valid")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion1 = FocalLoss(num_class=len(args.classes), alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')
    criterion2 = DiceLoss(num_classes=len(args.classes))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr)

    minimum_loss = float("inf")
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch : [{epoch+1:>03}|{args.epochs}], LR : {current_lr}")
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_acc, train_f1, train_dice = train(model, train_dataloader, optimizer, criterion1, criterion2, args.device, len(args.classes), args.semantic_loss_ignore_index)
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Accuracy', train_acc, epoch)
        writer.add_scalar('Training/F1 Score', train_f1, epoch)
        writer.add_scalar('Training/Mean Dice Coefficient', train_dice, epoch)
        print(f"Train Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}, Train F1 : {train_f1:.4f}, Train Dice : {train_dice:.4f}")

        valid_loss, valid_acc, valid_f1, valid_dice = valid(model, valid_dataloader, criterion1, criterion2, args.device, len(args.classes), args.semantic_loss_ignore_index)
        writer.add_scalar('Validation/Loss', valid_loss, epoch)
        writer.add_scalar('Validation/Accuracy', valid_acc, epoch)
        writer.add_scalar('Validation/F1 Score', valid_f1, epoch)
        writer.add_scalar('Validation/Mean Dice Coefficient', valid_dice, epoch)
        print(f"Valid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}, Valid F1 : {valid_f1:.4f}, Valid Dice : {valid_dice:.4f}")

        scheduler.step()
        if valid_loss < minimum_loss:
            print(f"Valid Loss improved {minimum_loss} --> {valid_loss}, model saved.")
            minimum_loss = valid_loss
            torch.save(model.state_dict(), f'{args.save_dir}/weights/best.pt')

        inference_callback(args.sample_img, model, feature_extractor, args, epoch, save_dir=args.save_dir)
        torch.save(model.state_dict(), f'{args.save_dir}/weights/last.pt')

    writer.close()
    torch.save(model.state_dict(), f'{args.save_dir}/weights/final.pt')

if __name__ == "__main__":
    args = Args("./config.yaml", is_train=True)
    set_seed(args.seed)
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.classes = BKAIDataset.CLASSES
    args.colormap = BKAIDataset.COLORMAP

    id2label = {idx: label for idx, label in enumerate(args.classes)}
    label2id = {label: idx for idx, label in id2label.items()}
    args.id2label = id2label
    args.label2id = label2id

    main(args)
