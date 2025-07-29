import os
import cv2
import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from dataset import NUM_CLASSES, IMAGE_SIZE, LABEL_MAP

EXPERIMENT_DIR = './experiment'
CHECKPOINT_PATH = os.path.join(EXPERIMENT_DIR, 'deeplabv3_best.pth')
VIDEO_PATH = 'chunked_video.mp4'  # Path to your input video
OUTPUT_VIDEO_PATH = 'output_segmented_chunked.mp4'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Color map for mask overlay
cmap = plt.get_cmap('tab10')
colors = (np.array([cmap(i)[:3] for i in range(NUM_CLASSES)]) * 255).astype(np.uint8)

# Preprocessing transform (no augmentation, just resize and normalize)
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def overlay_mask_on_image(image, mask, alpha=0.5):
    mask_rgb = np.zeros_like(image)
    for class_id in np.unique(mask):
        if class_id == 0:
            continue  # skip background
        mask_rgb[mask == class_id] = colors[class_id]
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    # Add label text at centroid of each region
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        label = [k for k, v in LABEL_MAP.items() if v == class_id]
        label = label[0] if label else str(class_id)
        # Find centroid
        region = (mask == class_id).astype(np.uint8)
        moments = cv2.moments(region)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return overlay

def main():
    # Load model
    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
    model = model.to(DEVICE)
    model.eval()

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open input video: {VIDEO_PATH}")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_count = 0
    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                # Preprocess
                input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
                # Inference
                output = model(input_tensor)['out']
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                # Resize mask to original frame size
                mask_resized = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                # Overlay mask
                overlay = overlay_mask_on_image(frame, mask_resized, alpha=0.5)
                out.write(overlay)
    finally:
        cap.release()
        out.release()
    print(f"Processed {frame_count} frames.")
    if frame_count > 0:
        print(f"Saved segmented video to {OUTPUT_VIDEO_PATH}")
    else:
        print("No frames processed. Output video not saved.")

if __name__ == "__main__":
    main() 