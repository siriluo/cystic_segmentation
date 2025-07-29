import os
import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import SegmentationDataset, get_dataloader

NUM_CLASSES = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '.'
IMAGE_DIR = os.path.join(DATA_DIR, 'images_val')
MASK_DIR = os.path.join(DATA_DIR, 'masks_val')
CHECKPOINT_PATH = 'deeplabv3_best.pth'

if __name__ == "__main__":
    dataloader = get_dataloader(IMAGE_DIR, MASK_DIR, batch_size=1, shuffle=False, num_workers=4)

    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    def compute_iou(pred, target, num_classes):
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        return ious

    ious = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            ious.append(compute_iou(preds.cpu(), masks.cpu(), NUM_CLASSES))
    ious = np.array(ious)
    mean_ious = np.nanmean(ious, axis=0)
    print('Per-class IoU:', mean_ious)
    print('Mean IoU:', np.nanmean(mean_ious)) 