import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from dataset import load_data_with_masks, train_val_split_samples, SegmentationDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Configurations
NUM_CLASSES = 8  # 7 classes + background
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './data'
EXPERIMENT_DIR = './experiment'
CHECKPOINT_PATH = os.path.join(EXPERIMENT_DIR, 'deeplabv3_best.pth')
IMAGE_SIZE = (512, 512)
LOG_PATH = os.path.join(EXPERIMENT_DIR, 'train_val_log.txt')
LOSS_PLOT_PATH = os.path.join(EXPERIMENT_DIR, 'loss_curve.png')

if __name__ == "__main__":
    # Create experiment directory
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # Load and split data
    samples = load_data_with_masks(DATA_ROOT)
    train_samples, val_samples = train_val_split_samples(samples, val_ratio=0.2)
    print(f"Number of training samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")
    # Increase augmentation strength by using a larger image size and more aggressive augmentations
    train_dataset = SegmentationDataset.from_samples(train_samples, augment=True, image_size=IMAGE_SIZE)
    val_dataset = SegmentationDataset.from_samples(val_samples, augment=False, image_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        best_loss = 0

    train_losses = []
    val_losses = []
    lrs = []
    with open(LOG_PATH, 'w') as log_file:
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}')
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Val Loss: {val_loss:.4f}')
            log_file.write(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}\n')
            log_file.flush()
            scheduler.step(val_loss)
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print('Model saved!')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {EARLY_STOPPING_PATIENCE} epochs.")
                    break

    # Plot loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    # Plot learning rate
    plt.figure()
    plt.plot(range(1, len(lrs)+1), lrs, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(EXPERIMENT_DIR, 'lr_curve.png'))
    plt.close()

    # Save a few sample validation predictions
    model.eval()
    cmap = plt.get_cmap('tab10')
    n_save = 5
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(DEVICE)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            imgs = images.cpu().numpy()
            gts = masks.cpu().numpy()
            for j in range(min(n_save, images.shape[0])):
                img = np.transpose(imgs[j], (1, 2, 0))
                img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img = np.clip(img, 0, 1)
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(img)
                axs[0].set_title('Input Image')
                axs[0].axis('off')
                axs[1].imshow(gts[j], cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
                axs[1].set_title('Ground Truth')
                axs[1].axis('off')
                axs[2].imshow(preds[j], cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
                axs[2].set_title('Prediction')
                axs[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(EXPERIMENT_DIR, f'val_result_{i}_{j}.png'))
                plt.close(fig)
            n_save -= images.shape[0]
            if n_save <= 0:
                break 