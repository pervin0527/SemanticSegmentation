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
from loss import FocalLoss, DiceLoss, JaccardLoss, compute_mean_dice_coefficient_score
from utils.utils import Args, inference_callback, plot_metrics, WarmupThenDecayScheduler

def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True

def valid(model, dataloader, criterion, device, ignore_idx=0):
    model.eval()
    accs, losses = [], []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Eval", leave=False)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False) ## mode="nearest"
            predicted = upsampled_logits.argmax(dim=1)
            
            loss = criterion(upsampled_logits, labels)
            losses.append(loss.item()) 

            mask = (labels != ignore_idx) # we don't include the background class in the accuracy calculation
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels, true_labels)
            accs.append(accuracy)
            
    avg_loss = sum(losses) / len(losses)
    avg_acc =  sum(accs) / len(accs)
    return avg_loss, avg_acc


def train(model, dataloader, optimizer, criterion, device, ignore_idx=0):
    model.train()
    accs, losses = [], []
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

        mask = (labels != ignore_idx) # we don't include the background class in the accuracy calculation
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels, true_labels)
        accs.append(accuracy)

    avg_loss = sum(losses) / len(losses)
    avg_acc =  sum(accs) / len(accs)

    return avg_loss, avg_acc
    

def main(args):
    writer = SummaryWriter(log_dir=f"{args.save_dir}/logs")
    # model_config = SegformerConfig.from_pretrained(args.pretrained_model_name,
    #                                                id2label=args.id2label, 
    #                                                label2id=args.label2id,
    #                                                num_labels=len(args.classes),
    #                                                image_size=args.img_size,
    #                                                num_encoder_blocks=args.num_encoder_blocks,
    #                                                drop_path_rate=args.drop_path_rate,
    #                                                hidden_dropout_prob=args.hidden_dropout_prob,
    #                                                classifier_dropout_prob=args.classifier_dropout_prob,
    #                                                attention_probs_dropout_prob=args.attention_probs_dropout_prob,
    #                                                semantic_loss_ignore_index=args.semantic_loss_ignore_index)

    model_config = SegformerConfig(
        id2label=args.id2label, 
        label2id=args.label2id,
        num_labels=len(args.classes),
        image_size=args.img_size,
        num_encoder_blocks=args.num_encoder_blocks,
        drop_path_rate=args.drop_path_rate,
        hidden_dropout_prob=args.hidden_dropout_prob,
        classifier_dropout_prob=args.classifier_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        semantic_loss_ignore_index=args.semantic_loss_ignore_index
    )
    model_config.save_pretrained(f'{args.save_dir}')
    model_config.save_pretrained(f'{args.save_dir}')
    
    # feature_extractor = SegformerImageProcessor.from_pretrained(args.pretrained_model_name, do_reduce_labels=args.do_reduce_labels)
    feature_extractor = SegformerImageProcessor(do_reduce_labels=args.do_reduce_labels)

    # model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_model_name,
    #                                                          config=model_config,
    #                                                          ignore_mismatched_sizes=True)

    model = SegformerForSemanticSegmentation(config=model_config)
    model.to(args.device)

    # train_dataset = VOCDataset(args, feature_extractor, image_set="trainval", year=2012)
    # valid_dataset = VOCDataset(args, feature_extractor, image_set="test", year=2007)
    train_dataset = BKAIDataset(args, feature_extractor, image_set="train")
    valid_dataset = BKAIDataset(args, feature_extractor, image_set="valid")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # criterion = FocalLoss(num_class=len(args.classes), alpha=args.focal_alpha, gamma=2, reduction='mean').to(args.device)
    criterion = DiceLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr)
    # scheduler = WarmupThenDecayScheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs, min_lr=args.min_lr, max_lr=args.max_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * args.epochs, eta_min=args.learning_rate / 100)

    epochs_no_improve = 0
    best_metric_score = 0.0
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch : [{epoch+1:>03}|{args.epochs}], LR : {current_lr}")
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, args.device, args.semantic_loss_ignore_index)
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Accuracy', train_acc, epoch)
        print(f"Train Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")

        valid_loss, valid_acc = valid(model, valid_dataloader, criterion, args.device, args.semantic_loss_ignore_index)
        writer.add_scalar('Validation/Loss', valid_loss, epoch)
        writer.add_scalar('Validation/Accuracy', valid_acc, epoch)
        print(f"Valid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}")
        scheduler.step()

        metric_score = compute_mean_dice_coefficient_score(model, valid_dataloader, args.device, len(args.classes))
        writer.add_scalar('Validation/Mean-Dice-Coefficient', metric_score, epoch)
        print(f"Valid Mean-Dice-Coefficient: {metric_score:.4f}")
        inference_callback(args.sample_img, model, feature_extractor, args, epoch, save_dir=f"{args.save_dir}")

        if metric_score > best_metric_score:
            best_metric_score = metric_score
            torch.save(model.state_dict(), f'{args.save_dir}/weights/best.pt')
            print(f"best metric improved, model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            print("Early stopping")
            break

    writer.close()
    torch.save(model.state_dict(), f'{args.save_dir}/weights/last.pt')

if __name__ == "__main__":
    args = Args("./config.yaml", is_train=True)
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