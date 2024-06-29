import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score

from torch.nn import functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig

from data.voc import VOCDataset
from data.bkai import BKAIDataset
from loss import FocalLoss, compute_mean_iou, compute_mean_dice_coefficient_score
from utils.utils import Args, inference_callback, plot_metrics, WarmupThenDecayScheduler


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

        loss = criterion(upsampled_logits, labels) ## focal loss
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
            
            loss = criterion(upsampled_logits, labels) ## focal loss
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
    full_dataset = BKAIDataset(args, feature_extractor, image_set="train")
    
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"Fold {fold+1}")

        fold_save_dir = os.path.join(args.save_dir, f'fold_{fold+1}')
        os.makedirs(fold_save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(fold_save_dir, 'logs'))

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_dataloader = DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.num_workers)

        model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_model_name,
                                                             config=model_config,
                                                             ignore_mismatched_sizes=True)
        model.to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion = FocalLoss(num_class=len(args.classes), alpha=args.focal_alpha, gamma=2, reduction='mean').to(args.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr)

        epochs_no_improve = 0
        best_metric_score = 0.0
        best_valid_loss = float('inf')
        for epoch in range(args.epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch : [{epoch+1:>03}|{args.epochs}], LR : {current_lr}")
            writer.add_scalar(f'Fold_{fold+1}/Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, args.device, args.semantic_loss_ignore_index)
            writer.add_scalar(f'Fold_{fold+1}/Training/Loss', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Training/Accuracy', train_acc, epoch)
            print(f"Train Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")

            valid_loss, valid_acc = valid(model, valid_dataloader, criterion, args.device, args.semantic_loss_ignore_index)
            writer.add_scalar(f'Fold_{fold+1}/Validation/Loss', valid_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Validation/Accuracy', valid_acc, epoch)
            print(f"Valid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}")
            scheduler.step()

            inference_callback(args.sample_img, model, feature_extractor, args, epoch)
            if (epoch + 1) % args.metric_step == 0:
                metric_score = compute_mean_dice_coefficient_score(model, valid_dataloader, args.device, len(args.classes))
                writer.add_scalar(f'Fold_{fold+1}/Validation/metric score', metric_score, epoch)
                print(f"Epoch [{epoch+1}/{args.epochs}] - metric score: {metric_score:.4f}")

                if metric_score > best_metric_score:
                    best_metric_score = metric_score
                    torch.save(model.state_dict(), os.path.join(fold_save_dir, 'weights', 'best.pt'))
                    print(f"best metric improved, model saved.")
                
            # Early Stopping Check
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.early_stop_patience:
                print("Early stopping")
                break

        writer.close()
        torch.save(model.state_dict(), os.path.join(fold_save_dir, 'weights', 'last.pt'))

if __name__ == "__main__":
    args = Args("./config.yaml", is_train=True)
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.seed = 42  # Seed for reproducibility

    args.classes = BKAIDataset.CLASSES
    args.colormap = BKAIDataset.COLORMAP

    id2label = {idx: label for idx, label in enumerate(args.classes)}
    label2id = {label: idx for idx, label in id2label.items()}
    args.id2label = id2label
    args.label2id = label2id

    main(args)