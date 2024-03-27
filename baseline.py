import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerImageProcessor

from loss import FocalLoss
from dataset import VOCDataset
from utils import Args, save_inference_mask


def valid(model, dataloader, criterion, epoch, args):
    model.eval()
    accs, losses = [], []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Eval", leave=False)):
            pixel_values = batch["pixel_values"].to(args.device)
            labels = batch["labels"].to(args.device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            losses.append(loss.item()) 

            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            mask = (labels != 255) # we don't include the background class in the accuracy calculation
            pred_labels = predicted[mask].detach().cpu().numpy()
            true_labels = labels[mask].detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels, true_labels)
            accs.append(accuracy)
            
    return sum(losses) / len(losses), sum(accs) / len(accs)


def train(model, dataloader, optimizer, criterion, args, epoch):
    model.train()
    accs, losses = [], []
    for idx, batch in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        pixel_values = batch["pixel_values"].to(args.device)
        labels = batch["labels"].to(args.device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)

        mask = (labels != 255) # we don't include the background class in the accuracy calculation
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels, true_labels)
        accs.append(accuracy)

    return sum(losses) / len(losses), sum(accs) / len(accs)


def main(args):
    writer = SummaryWriter(log_dir=f"{args.save_dir}/logs")
    # feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    feature_extractor = SegformerImageProcessor.from_pretrained(args.pretrained_model_name, do_reduce_labels=False)
    model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_model_name, 
                                                             ignore_mismatched_sizes=True,
                                                             num_labels=len(args.VOC_CLASSES), 
                                                             id2label=args.id2label, 
                                                             label2id=args.label2id,
                                                             reshape_last_stage=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = FocalLoss(num_classes=len(args.VOC_CLASSES)).to(args.device)

    train_dataset = VOCDataset(args, feature_extractor, image_set="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = VOCDataset(args, feature_extractor, image_set="val")
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch : [{epoch+1}|{args.epochs}], LR : {current_lr}")
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        train_acc, train_loss = train(model, train_dataloader, optimizer, criterion, args, epoch)
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Accuracy', train_acc, epoch)
        print(f"Train Acc : {train_acc:.4f}, Train Loss : {train_loss:.4f}")

        valid_acc, valid_loss = valid(model, valid_dataloader, criterion, epoch, args)
        writer.add_scalar('Validation/Loss', valid_loss, epoch)
        writer.add_scalar('Validation/Accuracy', valid_acc, epoch)
        print(f"Valid Acc : {valid_acc:.4f}, Valid Loss : {valid_loss:.4f}")

        save_inference_mask("./dog.jpg", model, feature_extractor, args, epoch)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{args.save_dir}/weights/best.pt')
            print(f"Valid Loss improved, model saved.")

    writer.close()


if __name__ == "__main__":
    args = Args("./baseline_cfg.yaml", is_train=True)
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    id2label = {idx: label for idx, label in enumerate(args.VOC_CLASSES)}
    label2id = {label: idx for idx, label in id2label.items()}
    args.id2label = id2label
    args.label2id = label2id

    main(args)