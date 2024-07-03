import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import random
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score

from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig

from data.voc import VOCDataset
from data.bkai import BKAIDataset
from loss import FocalLoss, dice_coefficient
from utils.utils import Args, inference_callback, WarmupThenDecayScheduler

def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True

def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True


def valid(model, dataloader, criterion, device, num_classes, ignore_idx=0):
    model.eval()
    losses = []
    dice_scores = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Eval", leave=False)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            
            loss = criterion(upsampled_logits, labels)
            losses.append(loss.item()) 

            mask = (labels != ignore_idx)
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            
            dice = dice_coefficient(pred_labels, true_labels, num_classes)
            dice_scores.append(dice)
            
    avg_loss = sum(losses) / len(losses)
    avg_dice = sum(dice_scores) / len(dice_scores)
    return avg_loss, avg_dice


def train(model, dataloader, optimizer, criterion, device, num_classes, ignore_idx=0):
    model.train()
    losses = []
    dice_scores = []
    for idx, batch in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)

        loss = criterion(upsampled_logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        mask = (labels != ignore_idx)
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        
        dice = dice_coefficient(pred_labels, true_labels, num_classes)
        dice_scores.append(dice)

    avg_loss = sum(losses) / len(losses)
    avg_dice = sum(dice_scores) / len(dice_scores)

    return avg_loss, avg_dice


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

    train_dataset = BKAIDataset(args, feature_extractor, image_set="train")
    valid_dataset = BKAIDataset(args, feature_extractor, image_set="valid")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = FocalLoss(num_class=len(args.classes), alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean').to(args.device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr)

    epochs_no_improve = 0
    max_dice_coefficient = 0.0
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch : [{epoch+1:>03}|{args.epochs}], LR : {current_lr}")
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_dice = train(model, train_dataloader, optimizer, criterion, args.device, len(args.classes), args.semantic_loss_ignore_index)
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Dice', train_dice, epoch)
        print(f"Train Loss : {train_loss:.4f}, Train Dice : {train_dice:.4f}")

        valid_loss, valid_dice = valid(model, valid_dataloader, criterion, args.device, len(args.classes), args.semantic_loss_ignore_index)
        writer.add_scalar('Validation/Loss', valid_loss, epoch)
        writer.add_scalar('Validation/Dice', valid_dice, epoch)
        print(f"Valid Loss : {valid_loss:.4f}, Valid Dice : {valid_dice:.4f}")

        inference_callback(args.sample_img, model, feature_extractor, args, epoch, save_dir=args.save_dir)
        if (epoch + 1) % args.metric_step == 0:

            metric_score = compute_mean_dice_coefficient_score(model, valid_dataloader, args.device, len(args.classes))
            writer.add_scalar('Validation/metric score', metric_score, epoch)
            print(f"Epoch [{epoch+1}/{args.epochs}] - metric score: {metric_score:.4f}")

            if metric_score > best_metric_score:
                best_metric_score = metric_score
                torch.save(model.state_dict(), f'{args.save_dir}/weights/best.pt')
                print(f"best metric improved, model saved.")

    writer.close()
    torch.save(model.state_dict(), f'{args.save_dir}/weights/last.pt')


if __name__ == "__main__":
    args = Args("./config.yaml", is_train=True)
    set_seed(args.seed)
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    args.classes = BKAIDataset.CLASSES
    args.colormap = BKAIDataset.COLORMAP

    id2label = {idx: label for idx, label in enumerate(args.classes)}
    label2id = {label: idx for idx, label in id2label.items()}
    args.id2label = id2label
    args.label2id = label2id

    main(args)