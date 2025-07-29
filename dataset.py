# Constants for import
NUM_CLASSES = 8  # 7 classes + background
IMAGE_SIZE = (512, 512)

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from skimage.draw import polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from matplotlib.colors import ListedColormap

class SegmentationAugmentation:
    """
    Apply the same geometric augmentation to both image and mask.
    Color augmentations are applied to the image only.
    """
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, mask):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = transforms.functional.hflip(image)
            mask = np.fliplr(mask)
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = transforms.functional.vflip(image)
            mask = np.flipud(mask)
        # Random rotation (0, 90, 180, 270 degrees)
        angle = int(np.random.choice([0, 90, 180, 270]))
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            mask = np.rot90(mask, k=angle // 90)
        # Color jitter (image only)
        image = self.color_jitter(image)
        # Resize
        image = transforms.functional.resize(image, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(Image.fromarray(mask), self.image_size, interpolation=transforms.InterpolationMode.NEAREST)
        mask = np.array(mask)
        # To tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image, torch.as_tensor(mask, dtype=torch.long)

def train_val_split_samples(samples, val_ratio=0.2, seed=42):
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split = int(len(samples) * (1 - val_ratio))
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    return train_samples, val_samples

class SegmentationDataset(Dataset):
    def __init__(self, data_root=None, samples=None, transform=None, augment=False, image_size=(512, 512)):
        if samples is not None:
            self.samples = samples
        elif data_root is not None:
            self.samples = load_data_with_masks(data_root)
        else:
            raise ValueError('Either data_root or samples must be provided')
        if augment:
            self.transform = SegmentationAugmentation(image_size=image_size)
        else:
            self.transform = transform or transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.augment = augment

    @classmethod
    def from_samples(cls, samples, **kwargs):
        return cls(samples=samples, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        mask = sample['mask']
        if self.augment:
            image, mask = self.transform(image, mask)
        else:
            image = self.transform(image)
            mask = Image.fromarray(mask)
            mask = transforms.functional.resize(mask, self.transform.transforms[0].size, interpolation=transforms.InterpolationMode.NEAREST)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

def get_dataloader(data_root, batch_size, shuffle=True, num_workers=4, augment=False, image_size=(512, 512)):
    dataset = SegmentationDataset(data_root, augment=augment, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Label mapping (update as needed)
LABEL_MAP = {
    'Gallbladder': 1,
    'Commonbileduct': 2,
    'Cysticduct': 3,
    'Instrument': 4,
    'liveredge': 5,
    'Cysticartery': 6,
    'LuviereSulcus': 7,
}

def load_data_with_masks(data_root):
    """
    Recursively loads all images and their annotation polygons from the data_root directory,
    converts polygons to masks, and returns a list of dicts with image path, mask, and label info.
    Handles overlapping polygons by label priority: Commonbileduct > Cysticduct > Cysticartery > others.
    Uses the region['region_attributes']['label'] for mask assignment.
    """
    # Priority: lower number = higher priority
    PRIORITY_MAP = {
        'Commonbileduct': 0,
        'Cysticduct': 1,
        'Cysticartery': 2,
    }
    samples = []
    for case_name in os.listdir(data_root):
        case_path = os.path.join(data_root, case_name)
        if not os.path.isdir(case_path):
            continue
        for subfolder in os.listdir(case_path):
            subfolder_path = os.path.join(case_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            # Find jpg and txt
            jpgs = [f for f in os.listdir(subfolder_path) if f.endswith('.jpg')]
            txts = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
            if not jpgs or not txts:
                continue
            img_path = os.path.join(subfolder_path, jpgs[0])
            txt_path = os.path.join(subfolder_path, txts[0])
            # Load image to get size
            with Image.open(img_path) as img:
                width, height = img.size
            # Load annotation
            with open(txt_path, 'r') as f:
                anno = json.load(f)
            # The annotation is a dict with the image filename as key
            key = os.path.basename(img_path)
            regions = anno[key]['regions']
            # Prepare regions as a list with priority
            region_list = []
            for region in regions.values():
                label = region['region_attributes'].get('label', None)
                if label is None or label not in LABEL_MAP:
                    continue
                priority = PRIORITY_MAP.get(label, 100)  # default low priority
                region_list.append((priority, label, region))
            # Sort by priority (lowest first)
            region_list.sort(key=lambda x: x[0])
            mask = np.zeros((height, width), dtype=np.uint8)
            for _, label, region in region_list:
                class_id = LABEL_MAP[label]
                shape = region['shape_attributes']
                if shape['name'] == 'polygon':
                    all_x = np.array(shape['all_points_x'])
                    all_y = np.array(shape['all_points_y'])
                    rr, cc = polygon(all_y, all_x, (height, width))
                    # Only update mask where it is still background (0) or lower priority
                    if label == 'Commonbileduct':
                        mask[rr, cc] = class_id  # always write
                    else:
                        # Only write where not already a higher-priority label
                        mask_area = mask[rr, cc]
                        overwrite = (mask_area == 0)
                        mask[rr[overwrite], cc[overwrite]] = class_id
            samples.append({'image_path': img_path, 'mask': mask, 'regions': regions})
    return samples

# test the function load_data_with_masks
if __name__ == "__main__":
    data_root = './data'
    print("Testing load_data_with_masks...")
    samples = load_data_with_masks(data_root)
    print(f"Loaded {len(samples)} samples from {data_root}")
    if samples:
        n_plot = min(5, len(samples))
        selected_samples = random.sample(samples, n_plot)
        # Define colors for each class (background + 7 classes)
        base_colors = plt.get_cmap('tab10').colors
        # Ensure we have enough colors (background + 7 classes)
        colors = [(0,0,0,1)] + list(base_colors[:7])
        class_ids = [0] + [LABEL_MAP[k] for k in LABEL_MAP]
        # Build a mapping from class id to color
        id_to_color = {cid: colors[i] for i, cid in enumerate(class_ids)}
        cmap = ListedColormap([id_to_color[cid] for cid in range(max(class_ids)+1)])
        id_to_label = {v: k for k, v in LABEL_MAP.items()}
        for i, sample in enumerate(selected_samples):
            print(f"Sample {i+1} image path: {sample['image_path']}")
            print(f"Sample {i+1} mask shape: {sample['mask'].shape}")
            print(f"Unique values in mask: {np.unique(sample['mask'])}")
            # Load the raw image
            raw_img = Image.open(sample['image_path'])
            # Create a side-by-side plot
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            # Show raw image
            axs[0].imshow(raw_img)
            axs[0].set_title('Raw Image')
            axs[0].axis('off')
            # Show mask
            im = axs[1].imshow(sample['mask'], cmap=cmap, vmin=0, vmax=max(class_ids))
            axs[1].set_title('Mask with Legend')
            axs[1].axis('off')
            # Build legend
            handles = []
            for class_id in np.unique(sample['mask']):
                if class_id == 0:
                    continue  # skip background
                label = id_to_label.get(class_id, f'Class {class_id}')
                color = id_to_color[class_id]
                patch = mpatches.Patch(color=color, label=label)
                handles.append(patch)
            axs[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'sample_{i+1}_mask_with_legend.png')
            plt.close(fig)

    print("\nTesting train/val split by sample...")
    train_samples, val_samples = train_val_split_samples(samples, val_ratio=0.2)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    train_dataset = SegmentationDataset.from_samples(train_samples, augment=True, image_size=(256, 256))
    val_dataset = SegmentationDataset.from_samples(val_samples, augment=False, image_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"[Train] Batch {batch_idx+1} - images shape: {images.shape}, masks shape: {masks.shape}")
        if batch_idx == 0:
            break
    for batch_idx, (images, masks) in enumerate(val_loader):
        print(f"[Val] Batch {batch_idx+1} - images shape: {images.shape}, masks shape: {masks.shape}")
        if batch_idx == 0:
            break
