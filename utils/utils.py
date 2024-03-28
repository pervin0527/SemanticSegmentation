import os
import cv2
import yaml
import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from torch.nn import functional as F


class Args:
    def __init__(self, hyp_dir, is_train=False):
        self.hyp_dir = hyp_dir  # hyp_dir을 인스턴스 변수로 저장
        self.hyps = self.load_config(hyp_dir)  # 설정 파일 로드
        for key, value in self.hyps.items():
            setattr(self, key, value)  # 객체에 key라는 변수를 생성하고, value를 값으로 할당함.
        
        if is_train:
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            base_save_dir = self.hyps.get("save_dir", "./saved_configs")
            self.save_dir = os.path.join(base_save_dir, current_time)
            
            self.make_dir(self.save_dir)
            self.save_config()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def save_config(self):
        save_path = os.path.join(self.save_dir, "config.yaml")
        with open(save_path, 'w') as file:
            yaml.dump(self.hyps, file)
        print(f"Config saved to {save_path}")

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(f"{path}/images")
            os.makedirs(f"{path}/weights")
            os.makedirs(f"{path}/logs")
            os.makedirs(f"{path}/test")

            print(f"{path} is generated.")
            
        else:
            print(f"{path} already exists.")


def save_inference_mask(image_path, model, feature_extractor, args, epoch):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.img_size, args.img_size))
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(args.device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()

    upsampled_logits = F.interpolate(
        logits,
        size=image.shape[:-1],  # (height, width)
        mode='bilinear',
        align_corners=False
    )

    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(args.COLORMAP):
        color_seg[seg == label, :] = np.array(color)

    mixed_img = (0.5 * image + 0.5 * color_seg).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(color_seg)
    axs[1].set_title("Predicted Mask")
    axs[1].axis('off')

    axs[2].imshow(mixed_img)
    axs[2].set_title("Mixed Image")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/images/EP{epoch:>04}.jpg")
    plt.close()


def compute_mean_iou(model, dataloader, device, num_labels, ignore_index=255):
    model.eval()
    metric = evaluate.load("mean_iou")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Mean IoU", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            # Update metric for each batch
            metric.add_batch(predictions=predicted, references=labels)

    # Compute the mean IoU over all batches
    metrics = metric.compute(num_labels=num_labels, ignore_index=ignore_index)
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    detailed_metrics = {
        "mean_iou": metrics["mean_iou"],
        **{f"accuracy_class_{i}": acc for i, acc in enumerate(per_category_accuracy)},
        **{f"iou_class_{i}": iou for i, iou in enumerate(per_category_iou)}
    }

    return detailed_metrics

def plot_metrics(detailed_metrics, class_names, metric_name, save_path):
    """
    detailed_metrics: compute_mean_iou에서 반환된 메트릭 딕셔너리
    class_names: 클래스 이름 리스트
    metric_name: 'accuracy' 또는 'iou' 중 하나
    save_path: 차트를 저장할 경로 (확장자 포함)
    """
    metric_values = [detailed_metrics[f"{metric_name}_class_{i}"] for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(class_names))
    plt.bar(y_pos, metric_values, align='center', alpha=0.7)
    plt.xticks(y_pos, class_names, rotation=45, ha="right")
    plt.ylabel(metric_name)
    plt.title(f'Class-wise {metric_name.upper()}')
    
    plt.tight_layout()  # 레이블이 잘리지 않도록 조정
    plt.savefig(save_path)
    plt.close()