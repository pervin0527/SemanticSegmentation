import os
import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

class Args:
    def __init__(self, hyp_dir, is_train=False, is_cv=False):
        self.hyp_dir = hyp_dir  # hyp_dir을 인스턴스 변수로 저장
        self.hyps = self.load_config(hyp_dir)  # 설정 파일 로드
        for key, value in self.hyps.items():
            setattr(self, key, value)  # 객체에 key라는 변수를 생성하고, value를 값으로 할당함.
        
        if is_train:
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            base_save_dir = self.hyps.get("save_dir", "./saved_configs")
            self.save_dir = os.path.join(base_save_dir, current_time)
            
            if not is_cv:
                self.make_dir(self.save_dir)
            else:
                os.makedirs(self.save_dir)
                
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


def inference_callback(image_path, model, feature_extractor, args, epoch, save_dir):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.img_size, args.img_size))
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(args.device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()

    upsampled_logits = F.interpolate(logits, size=image.shape[:-1], mode='bilinear', align_corners=False)

    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(args.colormap):
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
    plt.savefig(f"{save_dir}/images/EP{epoch:>04}.jpg")
    plt.close()


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

class WarmupThenDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr  # 최대 학습률을 추가합니다.
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.max_lr / self.base_lrs[0])
            return [base_lr * warmup_factor * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            decay_rate = (1 - ((self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            decay_rate = min(max(decay_rate, 0), 1)
            return [self.min_lr + (self.max_lr - self.min_lr) * decay_rate for _ in self.base_lrs]